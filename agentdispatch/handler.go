package agentdispatch

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"
)

// Message is the slice of an event-bus message this package needs. Narrowing it
// here keeps the decode / run / publish / ack policy unit-testable without a
// broker, which matters more than usual on this platform: a lane that broke
// silently cannot be diagnosed from the logs afterwards.
type Message interface {
	Data() []byte
	Attributes() map[string]string
	// DeliveryAttempt is 1 on first delivery and is populated by the client
	// only when the subscription carries a dead-letter policy. Zero means
	// unknown, which this package treats as "not the last attempt".
	DeliveryAttempt() int
	Ack()
	Nack()
}

// AgentRun executes the dispatched work and returns the agent's terminal text.
// The real implementation runs the ADK root agent in-process; tests inject a
// function. The returned string is passed through UNCHANGED — normalisation is
// the orchestrator's job, using the same mapper the HTTP path uses.
type AgentRun func(ctx context.Context, req Request) (string, error)

// Publisher publishes one completion.
type Publisher interface {
	Publish(ctx context.Context, topic string, body []byte, attrs map[string]string) error
}

// HandlerConfig wires one agent role's dispatch handler.
type HandlerConfig struct {
	AgentRole string
	Run       AgentRun
	Publisher Publisher
	// MaxDeliveryAttempts must match the subscription's deadLetterPolicy.
	// On the LAST attempt a failure is reported on the wire instead of nacked,
	// because the run on the other side is PARKED: a dead-lettered request with
	// no completion leaves the submission waiting forever.
	MaxDeliveryAttempts int
	// DedupeTTL bounds the in-process answer cache. Zero uses the default.
	DedupeTTL time.Duration
	Logger    *slog.Logger
}

const defaultDedupeTTL = 30 * time.Minute

// Handler applies the receive policy for one agent role.
type Handler struct {
	role        string
	run         AgentRun
	publisher   Publisher
	maxAttempts int
	log         *slog.Logger

	mu     sync.Mutex
	recent map[string]cachedAnswer
	ttl    time.Duration
}

type cachedAnswer struct {
	output string
	at     time.Time
}

// NewHandler builds a Handler. Panics on a missing collaborator rather than
// running as a no-op subscriber, which would ack every dispatch and answer none.
func NewHandler(cfg HandlerConfig) *Handler {
	if cfg.AgentRole == "" || cfg.Run == nil || cfg.Publisher == nil {
		panic("agentdispatch.NewHandler: AgentRole, Run and Publisher are required")
	}
	ttl := cfg.DedupeTTL
	if ttl <= 0 {
		ttl = defaultDedupeTTL
	}
	log := cfg.Logger
	if log == nil {
		log = slog.Default()
	}
	return &Handler{
		role: cfg.AgentRole, run: cfg.Run, publisher: cfg.Publisher,
		maxAttempts: cfg.MaxDeliveryAttempts, log: log,
		recent: map[string]cachedAnswer{}, ttl: ttl,
	}
}

// Handle applies the full receive policy to one message. It never panics out to
// the subscriber goroutine, and it acks only once the completion is genuinely
// published — acking earlier would strand a parked run whose work is finished
// but whose answer nobody ever receives.
func (h *Handler) Handle(ctx context.Context, msg Message) {
	var req Request
	if err := json.Unmarshal(msg.Data(), &req); err != nil {
		// ADR-254 D6 (arm a): where the routing fields are still readable the
		// parked run is told FAILED on its own lane; only an unreadable
		// message NACKs towards the DLQ, because nothing can be addressed.
		routing, readable := salvageRouting(msg.Data())
		if !readable {
			h.log.Error("agentdispatch.decode_failed", "err", err, "role", h.role, "readable", false)
			msg.Nack()
			return
		}
		h.log.Error("agentdispatch.decode_failed", "err", err, "role", h.role, "readable", true,
			"execution_id", routing.ExecutionID, "thread_id", routing.ThreadID)
		h.failAndAck(ctx, msg, routing, "decode_failed: "+err.Error())
		return
	}
	if req.AgentRole != "" && req.AgentRole != h.role {
		// Both OE binaries ship in one image, so a subscription pointed at the
		// wrong topic is a live misconfiguration risk. Answering it would
		// produce plausible output from the wrong agent. The request IS
		// readable, so its parked run hears FAILED on the lane it parked on
		// (ADR-254 D6 arm a) instead of waiting five deliveries for silence.
		h.log.Error("agentdispatch.role_mismatch",
			"want", h.role, "got", req.AgentRole, "execution_id", req.ExecutionID)
		h.failAndAck(ctx, msg, req, fmt.Sprintf(
			"role_mismatch: this subscriber serves %s but the request is addressed to %s",
			h.role, req.AgentRole))
		return
	}

	output, cached := h.cachedFor(req.IdempotencyKey)
	if !cached {
		var err error
		output, err = h.run(ctx, req)
		if err != nil {
			h.onRunError(ctx, msg, req, err)
			return
		}
		h.remember(req.IdempotencyKey, output)
	}

	if err := h.publish(ctx, req, Completion{
		Status:        StatusOK,
		OutputPayload: output,
	}); err != nil {
		// The work is done but the answer did not land. Nack: a redelivery
		// re-sends the cached answer without paying for the model again.
		h.log.Error("agentdispatch.publish_failed", "err", err,
			"execution_id", req.ExecutionID, "role", h.role)
		msg.Nack()
		return
	}
	msg.Ack()
	h.log.Info("agentdispatch.completed",
		"role", h.role, "execution_id", req.ExecutionID,
		"thread_id", req.ThreadID, "deduped", cached)
}

// onRunError decides between a retry and a terminal report.
func (h *Handler) onRunError(ctx context.Context, msg Message, req Request, runErr error) {
	attempt := msg.DeliveryAttempt()
	if isPermanent(runErr) {
		// ADR-254 D5/D6: a permanent fault (unknown discriminator, a gateway
		// FAILED_PRECONDITION such as surface_unstamped or companion_suspended)
		// cannot succeed on redelivery. Report it on the wire NOW so the parked
		// run ends with a status, and ack: four more deliveries through the
		// backoff ladder would only delay the same FAILED by minutes.
		h.log.Error("agentdispatch.run_failed_permanent", "err", runErr,
			"attempt", attempt, "execution_id", req.ExecutionID, "role", h.role)
		h.failAndAck(ctx, msg, req, permanentReason(runErr))
		return
	}
	final := h.maxAttempts > 0 && attempt >= h.maxAttempts
	if !final {
		// Retryable. Reporting a FAILED grade on a transient blip would end a
		// learner's run on infrastructure noise.
		h.log.Warn("agentdispatch.run_failed_retrying", "err", runErr,
			"attempt", attempt, "max", h.maxAttempts,
			"execution_id", req.ExecutionID, "role", h.role)
		msg.Nack()
		return
	}
	// Last attempt. The orchestrator's run is PARKED on this dispatch, so
	// dead-lettering in silence would leave the submission waiting forever.
	// Report the failure on the wire; the graph resumes and ends cleanly.
	h.log.Error("agentdispatch.run_failed_final", "err", runErr,
		"attempt", attempt, "execution_id", req.ExecutionID, "role", h.role)
	if err := h.publish(ctx, req, Completion{
		Status:       StatusFailed,
		ErrorMessage: runErr.Error(),
	}); err != nil {
		h.log.Error("agentdispatch.failure_publish_failed", "err", err,
			"execution_id", req.ExecutionID)
		msg.Nack()
		return
	}
	msg.Ack()
}

// failAndAck publishes a FAILED completion addressed as the REQUEST's own role
// and lane (its reply_topic, else that role's completion topic) and acks the
// message only once the completion is on the wire; a failed publish NACKs so
// the request is redelivered and, at the fifth attempt, dead-lettered.
func (h *Handler) failAndAck(ctx context.Context, msg Message, req Request, reason string) {
	role := strings.TrimSpace(req.AgentRole)
	if role == "" {
		role = h.role
	}
	if err := h.publishAs(ctx, role, req, Completion{Status: StatusFailed, ErrorMessage: reason}); err != nil {
		h.log.Error("agentdispatch.failure_publish_failed", "err", err,
			"execution_id", req.ExecutionID, "role", role)
		msg.Nack()
		return
	}
	msg.Ack()
}

// salvageRouting recovers the routing fields of a request whose typed decode
// failed (a field of the wrong JSON type, for instance). Readable means the
// parked run can be addressed: execution_id and thread_id are present and
// either a reply_topic or an agent_role names the lane.
func salvageRouting(data []byte) (Request, bool) {
	var loose map[string]any
	if err := json.Unmarshal(data, &loose); err != nil {
		return Request{}, false
	}
	str := func(k string) string {
		v, _ := loose[k].(string)
		return strings.TrimSpace(v)
	}
	r := Request{
		AgentRole: str("agent_role"), ExecutionID: str("execution_id"), ThreadID: str("thread_id"),
		IdempotencyKey: str("idempotency_key"), TenantID: str("tenant_id"), GCID: str("gcid"),
		ReplyTopic: str("reply_topic"), Traceparent: str("traceparent"), Tracestate: str("tracestate"),
	}
	readable := r.ExecutionID != "" && r.ThreadID != "" && (r.ReplyTopic != "" || r.AgentRole != "")
	return r, readable
}

func (h *Handler) publish(ctx context.Context, req Request, c Completion) error {
	return h.publishAs(ctx, h.role, req, c)
}

// publishAs publishes a completion authored as role (normally this handler's
// role; the request's own role for a FAILED on a mis-addressed request) to
// the request's reply_topic, else that role's completion topic.
func (h *Handler) publishAs(ctx context.Context, role string, req Request, c Completion) error {
	c.AgentRole = role
	c.ExecutionID = req.ExecutionID
	c.ThreadID = req.ThreadID
	c.IdempotencyKey = req.IdempotencyKey
	c.TenantID = req.TenantID
	c.Traceparent = req.Traceparent
	c.Tracestate = req.Tracestate
	c.CompletedAt = time.Now().UTC().Format(time.RFC3339Nano)

	body, err := json.Marshal(c)
	if err != nil {
		return err
	}
	topic := strings.TrimSpace(req.ReplyTopic)
	if topic == "" {
		topic = CompletionTopic(role)
	}
	attrs := map[string]string{
		// The completion's OWN key, distinct from the request's: the consumer
		// dedupes both through one table.
		"idempotency_key": req.CompletionKey(),
		"tenant_id":       req.TenantID,
		"gcid":            req.GCID,
		"agent_role":      role,
		"status":          c.Status,
		"event_topic":     topic,
	}
	// The traceparent rides the attributes AND the body. On HTTP it rode the
	// session state; if it rode neither here, one run would stop being one
	// trace across the hop.
	if req.Traceparent != "" {
		attrs["traceparent"] = req.Traceparent
	}
	if req.Tracestate != "" {
		attrs["tracestate"] = req.Tracestate
	}
	return h.publisher.Publish(ctx, topic, body, attrs)
}

func (h *Handler) cachedFor(key string) (string, bool) {
	if key == "" {
		return "", false
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	entry, ok := h.recent[key]
	if !ok || time.Since(entry.at) > h.ttl {
		return "", false
	}
	return entry.output, true
}

func (h *Handler) remember(key, output string) {
	if key == "" {
		return
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	now := time.Now()
	for k, v := range h.recent {
		if now.Sub(v.at) > h.ttl {
			delete(h.recent, k)
		}
	}
	h.recent[key] = cachedAnswer{output: output, at: now}
}
