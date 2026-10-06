package agentdispatch

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"

	"github.com/apollo-chora/chora-common/envelope"
	"github.com/apollo-chora/chora-common/eventbus"
	"github.com/apollo-chora/chora-common/tracing"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

// Env vars (never inlined per feedback_no_inline_config).
const (
	EnvNATSURL              = "NATS_URL"
	EnvDispatchSubscription = "AGENT_DISPATCH_SUBSCRIPTION"
	EnvDispatchEnabled      = "AGENT_DISPATCH_ENABLED"
	EnvMaxDeliveryAttempts  = "AGENT_DISPATCH_MAX_DELIVERY_ATTEMPTS"
)

// defaultMaxDeliveryAttempts MUST match the consumer's MaxDeliver ceiling,
// which is 5 across the platform. The handler uses it to decide when a
// failure stops being retryable and has to be reported on the wire instead,
// so a mismatch here means either a premature FAILED grade or a parked run
// that dead-letters in silence.
const defaultMaxDeliveryAttempts = 5

// ServeConfig wires one agent binary's dispatch subscriber.
type ServeConfig struct {
	// AgentRole is the DISPATCH role (oe_evaluate), not the ADK agent name
	// (oe_evaluator). The two differ on purpose.
	AgentRole string
	// ServiceName is the service name, used to derive the default
	// subscription so per-consumer attribution is legible at a glance.
	ServiceName string
	AppName     string
	RootAgent   agent.Agent
	Sessions    session.Service
	Plugins     runner.PluginConfig
	Logger      *slog.Logger

	// SessionKey, when set, makes the role CONVERSATIONAL (ADR-254 D6,
	// companion_chat): the dispatch runs in the session it names (user id,
	// session id) which is created on first sight, kept across dispatches and
	// never deleted, each dispatch merging its payload into the session state
	// as a state delta. Nil keeps the stateless per-dispatch session policy
	// (one fresh session per execution id, cleared on redelivery) every other
	// crew runs on. Returning ok=false falls back to that stateless policy for
	// that request.
	SessionKey func(req Request) (userID, sessionID string, ok bool)
	// UserMessage, when set, supplies the user turn text the runner feeds
	// the agent for this request (the learner's message on a chat turn). Nil
	// or "" sends the fleet's "BEGIN" trigger: the real prompt rides in
	// session state for every non-conversational role.
	UserMessage func(req Request) string
}

// Enabled reports whether this binary should start its dispatch subscriber.
//
// Off by default. ADR-253 D6 keeps rollback as "select the HTTP executor and
// redeploy", and that is only true while the HTTP surface keeps serving, so
// the two transports coexist and the event lane is opt-in per deployment.
func Enabled() bool {
	v := strings.TrimSpace(os.Getenv(EnvDispatchEnabled))
	return v == "1" || strings.EqualFold(v, "true")
}

// Serve runs the dispatch subscriber until ctx is cancelled.
//
// Intended to run in a goroutine BESIDE the ADK launcher's HTTP server, not
// instead of it: the agent answers both transports for the duration of the
// cutover, which is what makes the rollback in ADR-253 D6 real rather than
// nominal.
//
// Transport is NATS JetStream via chora-common/eventbus (the broker-neutral
// event bus): the request topic is a JetStream subject on the shared
// CHORA_EVENTS stream, and completions are published back onto the reply
// (completion) subject with the canonical event envelope as NATS headers.
// The consume loop is sequential — one dispatch at a time — which holds the
// line against a burst of concurrent calls against shared model quota.
func Serve(ctx context.Context, cfg ServeConfig) error {
	log := cfg.Logger
	if log == nil {
		log = slog.Default()
	}
	if cfg.AgentRole == "" || cfg.RootAgent == nil || cfg.Sessions == nil {
		return errors.New("agentdispatch.Serve: AgentRole, RootAgent and Sessions are required")
	}
	url := strings.TrimSpace(os.Getenv(EnvNATSURL))
	if url == "" {
		return fmt.Errorf("agentdispatch.Serve: %s is unset", EnvNATSURL)
	}
	subscription := strings.TrimSpace(os.Getenv(EnvDispatchSubscription))
	if subscription == "" {
		subscription = RequestSubscription(cfg.ServiceName, cfg.AgentRole)
	}

	appName := cfg.AppName
	if appName == "" {
		appName = cfg.AgentRole
	}
	// The SAME session service instance goes to the runner and to the agent
	// run: a runner pointed at a different store would find no session state
	// and the agent would render an empty [TASK] block.
	r, err := runner.New(runner.Config{
		AppName:        appName,
		Agent:          cfg.RootAgent,
		SessionService: cfg.Sessions,
		PluginConfig:   cfg.Plugins,
	})
	if err != nil {
		return fmt.Errorf("agentdispatch.Serve: runner.New: %w", err)
	}

	bus, err := eventbus.NewJetStream(eventbus.JetStreamConfig{URL: url})
	if err != nil {
		return fmt.Errorf("agentdispatch.Serve: eventbus.NewJetStream(%s): %w", url, err)
	}
	defer func() { _ = bus.Close() }()

	handler := NewHandler(HandlerConfig{
		AgentRole: cfg.AgentRole,
		Run: RunnerAgentRunWithPolicy(r, cfg.Sessions, appName, RunPolicy{
			SessionKey: cfg.SessionKey, UserMessage: cfg.UserMessage,
		}),
		Publisher:           &busPublisher{bus: bus, sourceService: cfg.ServiceName},
		MaxDeliveryAttempts: maxDeliveryAttemptsFromEnv(log),
		Logger:              log,
	})

	log.Info("agentdispatch.serving",
		"role", cfg.AgentRole, "bus", url, "subscription", subscription)

	if err := bus.Subscribe(ctx, eventbus.ConsumerConfig{
		Name:       subscription,
		Subject:    RequestTopic(cfg.AgentRole),
		MaxDeliver: maxDeliveryAttemptsFromEnv(log),
	}, func(ctx context.Context, m eventbus.Message) error {
		bm := &busMessage{msg: m}
		handler.Handle(ctx, bm)
		// The event bus settles by the handler's return: nil acks, error
		// nacks towards redelivery. Handle always settles exactly once.
		switch {
		case !bm.settled:
			return errors.New("agentdispatch: handler did not settle the message")
		case bm.acked:
			return nil
		default:
			return errNacked
		}
	}); err != nil {
		return err
	}

	// Subscribe starts the consume loop asynchronously and returns at once, so
	// this function must not return yet: the deferred bus.Close() above would
	// tear the connection down the instant the subscription came up, leaving a
	// consumer that exists on the broker but never pulls (no deliveries, no
	// errors, no log line). Block until the process is asked to stop.
	<-ctx.Done()
	return nil
}

// errNacked is the redelivery signal the subscribe wrapper returns when the
// handler settled a message with Nack.
var errNacked = errors.New("agentdispatch: message nacked")

// busMessage adapts an eventbus.Message to the narrow Message interface.
// Ack/Nack record the settlement; the subscribe wrapper translates it into
// the eventbus ack/nack contract.
type busMessage struct {
	msg     eventbus.Message
	settled bool
	acked   bool
}

func (c *busMessage) Data() []byte { return c.msg.Payload }

// Attributes projects the event envelope onto the attribute map shape the
// interface declares. The dispatch handler reads its routing fields from the
// decoded body, so this exists for logging + future consumers.
func (c *busMessage) Attributes() map[string]string {
	env := c.msg.Envelope
	attrs := map[string]string{
		"event_id":        env.EventID,
		"idempotency_key": env.IdempotencyKey,
		"tenant_id":       env.TenantID,
		"gcid":            env.GCID,
		"source_service":  env.SourceService,
		"source_project":  env.SourceProject,
	}
	if env.Traceparent != "" {
		attrs["traceparent"] = env.Traceparent
	}
	if env.Tracestate != "" {
		attrs["tracestate"] = env.Tracestate
	}
	return attrs
}

// DeliveryAttempt is 1 on first delivery and grows on redelivery. Zero means
// unknown, which this package treats as "not the last attempt".
func (c *busMessage) DeliveryAttempt() int { return int(c.msg.DeliveryAttempt) }

func (c *busMessage) Ack()  { c.settled = true; c.acked = true }
func (c *busMessage) Nack() { c.settled = true; c.acked = false }

// busPublisher adapts the eventbus bus to the Publisher interface. The
// completion's attributes become the canonical event envelope (NATS headers)
// so the orchestrator's consumer reads tenant/gcid/traceparent without
// decoding the payload.
type busPublisher struct {
	bus           eventbus.Bus
	sourceService string
}

func (p *busPublisher) Publish(ctx context.Context, topic string, body []byte, attrs map[string]string) error {
	ctx = tracing.WithTenantID(ctx, attrs["tenant_id"])
	ctx = tracing.WithGCID(ctx, attrs["gcid"])
	ctx = tracing.WithTraceparent(ctx, attrs["traceparent"])
	env := envelope.Build(ctx, envelope.BuildOpts{
		EventType:      topic,
		SchemaVersion:  1,
		SourceProject:  "chora",
		SourceService:  p.sourceService,
		IdempotencyKey: attrs["idempotency_key"],
	})
	// JetStream publish is synchronous to the server ack. The caller acks the
	// REQUEST only after this returns, so a fire-and-forget publish here would
	// let a completion be lost while its request was acked, parking the run
	// forever.
	if err := p.bus.Publish(ctx, topic, env, body); err != nil {
		return fmt.Errorf("agentdispatch: publish to %s: %w", topic, err)
	}
	return nil
}

func maxDeliveryAttemptsFromEnv(log *slog.Logger) int {
	raw := strings.TrimSpace(os.Getenv(EnvMaxDeliveryAttempts))
	if raw == "" {
		return defaultMaxDeliveryAttempts
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		log.Warn("agentdispatch.invalid_max_delivery_attempts",
			"value", raw, "default", defaultMaxDeliveryAttempts)
		return defaultMaxDeliveryAttempts
	}
	return n
}
