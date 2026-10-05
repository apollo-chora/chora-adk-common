// Package agentdispatch is the agent side of the event-dispatch pair
// ratified in ADR-253.
//
// Before ADR-253 an agent was reached by an awaited HTTP POST: the Python
// orchestrator opened a session, streamed a query, and held the call open for
// the whole model latency. This package replaces that hop with a subscriber
// INSIDE the agent binary. It consumes a dispatch request, runs the agent
// in-process through the ADK runner (the same root agent and the same plugin
// chain the HTTP launcher serves), and publishes a completion the orchestrator
// resumes its parked graph from.
//
// ⚠ This is deliberately NOT a subscribe-and-forward shim. A shim that posted
// to the agent's own local HTTP surface would be far cheaper, but it would
// leave the agent request-driven behind an event-driven wire, which is a weaker
// claim than ADR-253 makes and would have to be labelled as such everywhere the
// design is described. The owner ruled the native subscriber on 2026-08-20.
//
// The wire contract mirrors the Python producer, which is the source of
// truth for it. The two must agree exactly: a wrong topic name subscribes to
// nothing and a wrong session-state key renders an empty [TASK] block, so the
// agent would grade an answer it cannot see. Both failures are silent.
package agentdispatch

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Completion status discriminator. ADR-253 D4 carries the outcome INSIDE the
// completion payload rather than on separate success and failure topics, so a
// failed dispatch travels the same wire and resumes the same parked run.
const (
	StatusOK     = "OK"
	StatusFailed = "FAILED"
)

const topicPrefix = "chora.ai_kernel.agent_dispatch"

// RequestTopic is where the orchestrator publishes work for a role.
func RequestTopic(agentRole string) string {
	return fmt.Sprintf("%s.%s_requested.v1", topicPrefix, agentRole)
}

// CompletionTopic is where this agent publishes its answer.
func CompletionTopic(agentRole string) string {
	return fmt.Sprintf("%s.%s_completed.v1", topicPrefix, agentRole)
}

// RequestSubscription is the agent's own subscription on the request topic.
// Named after the consuming service so per-subscription IAM (which inherits
// nothing) is legible at a glance in the console.
func RequestSubscription(serviceName, agentRole string) string {
	return fmt.Sprintf("%s.agent-dispatch-%s-requested",
		serviceName, strings.ReplaceAll(agentRole, "_", "-"))
}

// Request is one dispatch as it arrives on the wire. Field names are the JSON
// keys the Python producer writes; do not rename without changing both sides.
type Request struct {
	AgentRole        string `json:"agent_role"`
	ExecutionID      string `json:"execution_id"`
	TenantID         string `json:"tenant_id"`
	GCID             string `json:"gcid"`
	AGID             string `json:"agid"`
	ThreadID         string `json:"thread_id"`
	IdempotencyKey   string `json:"idempotency_key"`
	PromptTemplateID string `json:"prompt_template_id"`
	InputPayload     string `json:"input_payload"`
	ReplyTopic       string `json:"reply_topic"`
	Traceparent      string `json:"traceparent"`
	Tracestate       string `json:"tracestate"`
	RequestedAt      string `json:"requested_at"`
}

// Completion is the agent's answer. The orchestrator's executor maps
// OutputPayload with the SAME mapper the HTTP path uses, so this side does not
// normalise the agent's JSON and must not: re-deriving the token split here
// would quietly rewrite every pipeline_trace row and every O+ token tile.
type Completion struct {
	AgentRole      string `json:"agent_role"`
	ExecutionID    string `json:"execution_id"`
	ThreadID       string `json:"thread_id"`
	IdempotencyKey string `json:"idempotency_key"`
	Status         string `json:"status"`
	OutputPayload  string `json:"output_payload"`
	ErrorMessage   string `json:"error_message,omitempty"`
	Traceparent    string `json:"traceparent,omitempty"`
	Tracestate     string `json:"tracestate,omitempty"`
	TenantID       string `json:"tenant_id,omitempty"`
	CompletedAt    string `json:"completed_at,omitempty"`
}

// CompletionKey is the completion's OWN idempotency key.
//
// It must differ from the request's: both sides dedupe through one
// idempotency_keys table on the orchestrator, so reusing the request key would
// make the first completion look like an already-seen request and the run would
// park forever.
func (r Request) CompletionKey() string {
	return r.IdempotencyKey + ".completed"
}

// StateKeyDispatchExecutionID is the session-state key under which
// SessionState records the dispatch's execution id (read by
// terminationplugin to name the run on agent_terminated).
const StateKeyDispatchExecutionID = "dispatch_execution_id"

// SessionState builds the ADK session state for this request.
//
// Mirrors the Python _build_session_state OE branch exactly: the base identity
// keys the mana plugin requires, the W3C trace context from the ENVELOPE, and
// then every key the crew node threaded through input_payload. The crew node
// controls session state entirely, which is why this is a merge and not a
// hand-picked field list.
func (r Request) SessionState() (map[string]any, error) {
	tenantID := strings.TrimSpace(r.TenantID)
	if tenantID == "" {
		// The mana plugin refuses a session without tenant_id + user_gcid, and
		// an untenanted run cannot be metered or attributed. Refuse here so the
		// failure names its cause instead of surfacing as a plugin rejection.
		return nil, fmt.Errorf("agentdispatch: request %q carries no tenant_id",
			r.ExecutionID)
	}
	gcid := strings.TrimSpace(r.GCID)
	if gcid == "" {
		gcid = "qgen-anon:" + tenantID
	}

	var input map[string]any
	if payload := strings.TrimSpace(r.InputPayload); payload != "" {
		if err := json.Unmarshal([]byte(payload), &input); err != nil {
			return nil, fmt.Errorf("agentdispatch: input_payload for %q is not a "+
				"JSON object: %w", r.ExecutionID, err)
		}
	}

	state := make(map[string]any, len(input)+5)
	for k, v := range input {
		// Handled explicitly below from the envelope, which is authoritative: it
		// is what the broker carried and what the parked run will resume on. A
		// stale copy in the body must not win.
		switch k {
		case "gcid", "author_gcid", "traceparent", "tracestate", "dispatch_idempotency_key", "dispatch_execution_id":
			continue
		}
		state[k] = v
	}
	state["tenant_id"] = tenantID
	state["user_gcid"] = gcid
	state["author_gcid"] = gcid
	// ADR-254 D7 / R22: the dispatch key rides agent -> gateway, read from
	// session state by the tenant-propagation plugin and stamped on every
	// Invoke, so a redelivered dispatch is a keyed debit claim and bills once.
	if key := strings.TrimSpace(r.IdempotencyKey); key != "" {
		state["dispatch_idempotency_key"] = key
	}
	// The dispatch's execution id rides in state so a plugin that names the
	// run (terminationplugin's agent_terminated ExecutionID) can correlate on
	// the DISPATCH even when the session is a conversation that outlives it
	// (companion_chat: Session().ID() is the conversation id there).
	if id := strings.TrimSpace(r.ExecutionID); id != "" {
		state[StateKeyDispatchExecutionID] = id
	}
	if tp := strings.TrimSpace(r.Traceparent); tp != "" {
		state["traceparent"] = tp
		if ts := strings.TrimSpace(r.Tracestate); ts != "" {
			state["tracestate"] = ts
		}
	}
	return state, nil
}
