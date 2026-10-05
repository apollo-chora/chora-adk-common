package agentdispatch

import (
	"encoding/json"
	"testing"
)

// The wire contract is shared with the Python orchestrator, which is the
// producer. These tests pin the Go side to the SAME strings the producer emits,
// because a mismatch here is silent in both directions: a wrong topic name
// subscribes to nothing, and a wrong session-state key renders an empty [TASK]
// block and the agent grades an answer it cannot see.

func TestCompletionTopicMatchesTheProducerContract(t *testing.T) {
	for role, want := range map[string]string{
		"oe_evaluate": "chora.ai_kernel.agent_dispatch.oe_evaluate_completed.v1",
		"oe_moderate": "chora.ai_kernel.agent_dispatch.oe_moderate_completed.v1",
	} {
		if got := CompletionTopic(role); got != want {
			t.Errorf("CompletionTopic(%q) = %q, want %q", role, got, want)
		}
	}
}

func TestRequestSubscriptionMatchesTheProvisionedName(t *testing.T) {
	want := "chora-oe-evaluator.agent-dispatch-oe-evaluate-requested"
	if got := RequestSubscription("chora-oe-evaluator", "oe_evaluate"); got != want {
		t.Errorf("RequestSubscription = %q, want %q", got, want)
	}
}

func TestSessionStateMirrorsTheHTTPExecutorContract(t *testing.T) {
	// The Python _build_session_state OE branch: base {tenant_id, user_gcid,
	// author_gcid}, then EVERY input_payload key except gcid / author_gcid /
	// traceparent / tracestate, which are handled explicitly.
	req := Request{
		AgentRole:   "oe_evaluate",
		TenantID:    "11111111-1111-7111-8111-111111111111",
		GCID:        "learner-1",
		Traceparent: "00-" + "a" + "-b-01",
		Tracestate:  "x=1",
		InputPayload: `{"mode":"evaluate","prompt":"Explain entropy",
			"rubric_json":"[]","learner_answer":"because","points_possible":5,
			"gcid":"learner-1","traceparent":"SHOULD-NOT-OVERRIDE"}`,
	}

	state, err := req.SessionState()
	if err != nil {
		t.Fatalf("SessionState: %v", err)
	}
	for k, want := range map[string]any{
		"tenant_id":   "11111111-1111-7111-8111-111111111111",
		"user_gcid":   "learner-1",
		"author_gcid": "learner-1",
		"mode":        "evaluate",
		"prompt":      "Explain entropy",
		"traceparent": "00-a-b-01",
		"tracestate":  "x=1",
	} {
		if got := state[k]; got != want {
			t.Errorf("state[%q] = %#v, want %#v", k, got, want)
		}
	}
	// The envelope's trace context wins over a stale copy in the body: the
	// envelope is what the broker carries and what the consumer resumed on.
	if state["traceparent"] == "SHOULD-NOT-OVERRIDE" {
		t.Error("body traceparent overrode the envelope's")
	}
	if _, present := state["gcid"]; present {
		t.Error("raw gcid leaked into session state; the contract key is user_gcid")
	}
}

func TestSessionStateThreadsThePromptOverride(t *testing.T) {
	// ADR-197 M-B.2 — the composer's overrideOr(segment_id, embedded) merge
	// reads these three PINNED keys. Dropping them silently reverts every
	// tenant's prompt override to the embedded default.
	req := Request{
		AgentRole: "oe_evaluate", TenantID: "t1", GCID: "g1",
		InputPayload: `{"prompt_overrides_json":"{\"role\":\"be strict\"}",
			"resolved_prompt_version":"v7","prompt_source":"tenant_override"}`,
	}
	state, err := req.SessionState()
	if err != nil {
		t.Fatalf("SessionState: %v", err)
	}
	for _, k := range []string{"prompt_overrides_json", "resolved_prompt_version", "prompt_source"} {
		if state[k] == nil || state[k] == "" {
			t.Errorf("prompt-override key %q was dropped from session state", k)
		}
	}
}

func TestAMissingTenantIsRefused(t *testing.T) {
	// A dispatch with no tenant cannot be metered by the mana plugin and
	// cannot be attributed. Refuse it rather than run it as nobody.
	req := Request{AgentRole: "oe_evaluate", GCID: "g1", InputPayload: "{}"}
	if _, err := req.SessionState(); err == nil {
		t.Fatal("SessionState accepted a request with no tenant_id")
	}
}

func TestCompletionRoundTripsAsJSON(t *testing.T) {
	c := Completion{
		AgentRole: "oe_evaluate", ExecutionID: "sub-1:tsq-1:1", ThreadID: "sub-1",
		IdempotencyKey: "agent_dispatch.oe_evaluate.sub-1:tsq-1:1",
		Status:         StatusOK, OutputPayload: `{"comment":"ok"}`,
	}
	raw, err := json.Marshal(c)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back map[string]any
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	// snake_case, because the Python consumer reads these keys verbatim.
	for _, k := range []string{"agent_role", "execution_id", "thread_id",
		"idempotency_key", "status", "output_payload"} {
		if _, ok := back[k]; !ok {
			t.Errorf("completion is missing wire key %q", k)
		}
	}
}

func TestSessionStateCarriesTheDispatchIdempotencyKey(t *testing.T) {
	// ADR-254 D7 / R22: the agent forwards the dispatch key to the gateway, so
	// it must be in session state where the tenant-propagation plugin reads
	// per-request values; a key from the body must not shadow the envelope's.
	r := Request{
		ExecutionID: "sub-1:q-1:1", TenantID: "11111111-1111-7111-8111-111111111111",
		IdempotencyKey: "agent_dispatch.oe_evaluate.sub-1:q-1:1",
		InputPayload:   `{"mode":"evaluate","dispatch_idempotency_key":"stale-from-body"}`,
	}
	state, err := r.SessionState()
	if err != nil {
		t.Fatal(err)
	}
	if got := state["dispatch_idempotency_key"]; got != r.IdempotencyKey {
		t.Errorf("dispatch_idempotency_key = %v, want the envelope key %q", got, r.IdempotencyKey)
	}
}

func TestSessionStateCarriesTheDispatchExecutionID(t *testing.T) {
	// The execution id rides in state so a plugin that names the run
	// (terminationplugin) can correlate on the DISPATCH even when the session
	// is a conversation that outlives it; a stale copy in the body never wins.
	r := Request{
		AgentRole: "companion_chat", ExecutionID: "exec-77", TenantID: "t1", GCID: "g1",
		InputPayload: `{"turn_kind":"typed","dispatch_execution_id":"stale-from-body"}`,
	}
	state, err := r.SessionState()
	if err != nil {
		t.Fatal(err)
	}
	if got := state[StateKeyDispatchExecutionID]; got != "exec-77" {
		t.Errorf("%s = %v, want the envelope's execution id", StateKeyDispatchExecutionID, got)
	}
	// Absent on the envelope: not stamped (and the body's copy still dropped).
	r.ExecutionID = "  "
	state, _ = r.SessionState()
	if _, ok := state[StateKeyDispatchExecutionID]; ok {
		t.Errorf("an empty execution id must not be stamped: %v", state[StateKeyDispatchExecutionID])
	}
}
