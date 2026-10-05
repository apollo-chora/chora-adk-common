package agentdispatch

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

type fakeMsg struct {
	data    []byte
	attrs   map[string]string
	attempt int
	acked   bool
	nacked  bool
}

func (m *fakeMsg) Data() []byte                  { return m.data }
func (m *fakeMsg) Attributes() map[string]string { return m.attrs }
func (m *fakeMsg) DeliveryAttempt() int          { return m.attempt }
func (m *fakeMsg) Ack()                          { m.acked = true }
func (m *fakeMsg) Nack()                         { m.nacked = true }

type fakePublisher struct {
	published []struct {
		topic string
		body  Completion
		attrs map[string]string
	}
	err error
}

func (p *fakePublisher) Publish(_ context.Context, topic string, body []byte, attrs map[string]string) error {
	if p.err != nil {
		return p.err
	}
	var c Completion
	if err := json.Unmarshal(body, &c); err != nil {
		return err
	}
	p.published = append(p.published, struct {
		topic string
		body  Completion
		attrs map[string]string
	}{topic, c, attrs})
	return nil
}

func req(t *testing.T) *fakeMsg {
	t.Helper()
	r := Request{
		AgentRole: "oe_evaluate", ExecutionID: "sub-1:tsq-1:1",
		TenantID: "11111111-1111-7111-8111-111111111111", GCID: "g1",
		ThreadID: "sub-1", IdempotencyKey: "agent_dispatch.oe_evaluate.sub-1:tsq-1:1",
		InputPayload: `{"mode":"evaluate"}`,
		ReplyTopic:   CompletionTopic("oe_evaluate"),
		Traceparent:  "00-aaaa-bbbb-01",
	}
	raw, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	return &fakeMsg{data: raw, attrs: map[string]string{
		"idempotency_key": r.IdempotencyKey, "tenant_id": r.TenantID,
	}, attempt: 1}
}

func handler(run AgentRun, pub Publisher) *Handler {
	return NewHandler(HandlerConfig{
		AgentRole: "oe_evaluate", Run: run, Publisher: pub, MaxDeliveryAttempts: 5,
	})
}

func okRun(out string) AgentRun {
	return func(context.Context, Request) (string, error) { return out, nil }
}

func TestASuccessfulRunPublishesACompletionAndAcks(t *testing.T) {
	pub := &fakePublisher{}
	msg := req(t)

	handler(okRun(`{"comment":"ok"}`), pub).Handle(context.Background(), msg)

	if !msg.acked || msg.nacked {
		t.Fatalf("ack=%v nack=%v, want ack", msg.acked, msg.nacked)
	}
	if len(pub.published) != 1 {
		t.Fatalf("published %d completions, want 1", len(pub.published))
	}
	got := pub.published[0]
	if got.topic != CompletionTopic("oe_evaluate") {
		t.Errorf("topic = %q", got.topic)
	}
	if got.body.Status != StatusOK || got.body.ThreadID != "sub-1" {
		t.Errorf("completion = %+v", got.body)
	}
	if got.body.OutputPayload != `{"comment":"ok"}` {
		t.Errorf("output_payload = %q", got.body.OutputPayload)
	}
	// The completion key must differ from the request's or the orchestrator's
	// shared idempotency table swallows the first completion.
	if got.attrs["idempotency_key"] != "agent_dispatch.oe_evaluate.sub-1:tsq-1:1.completed" {
		t.Errorf("attr idempotency_key = %q", got.attrs["idempotency_key"])
	}
	// One run is one trace across the hop: the traceparent must ride BOTH the
	// attributes and the body (ADR-253 D4).
	if got.attrs["traceparent"] != "00-aaaa-bbbb-01" || got.body.Traceparent != "00-aaaa-bbbb-01" {
		t.Errorf("traceparent lost: attrs=%q body=%q", got.attrs["traceparent"], got.body.Traceparent)
	}
}

func TestATransientFailureNacksForRedelivery(t *testing.T) {
	// A retryable failure must NOT be reported as a FAILED grade; that would
	// end the learner's run on a blip.
	pub := &fakePublisher{}
	msg := req(t)
	fail := func(context.Context, Request) (string, error) {
		return "", errors.New("model gateway 503")
	}

	handler(fail, pub).Handle(context.Background(), msg)

	if !msg.nacked || msg.acked {
		t.Fatalf("ack=%v nack=%v, want nack", msg.acked, msg.nacked)
	}
	if len(pub.published) != 0 {
		t.Fatalf("published a completion on a retryable failure: %+v", pub.published)
	}
}

func TestTheLastDeliveryAttemptPublishesAFailedCompletion(t *testing.T) {
	// The run is PARKED. If the last attempt only nacked, the message would
	// dead-letter and the submission would sit in PENDING_OE_GRADING forever.
	// The terminal attempt therefore reports the failure ON THE WIRE so the
	// graph resumes and ends cleanly.
	pub := &fakePublisher{}
	msg := req(t)
	msg.attempt = 5
	fail := func(context.Context, Request) (string, error) {
		return "", errors.New("model gateway 503")
	}

	handler(fail, pub).Handle(context.Background(), msg)

	if len(pub.published) != 1 {
		t.Fatalf("published %d completions on the final attempt, want 1", len(pub.published))
	}
	if pub.published[0].body.Status != StatusFailed {
		t.Errorf("status = %q, want %q", pub.published[0].body.Status, StatusFailed)
	}
	if pub.published[0].body.ErrorMessage == "" {
		t.Error("FAILED completion carries no error_message")
	}
	if !msg.acked {
		t.Error("final attempt did not ack after reporting the failure")
	}
}

func TestARedeliveryDoesNotPayForASecondModelCall(t *testing.T) {
	// In-process dedupe. Bounded by the pod lifetime by design: the agent has no
	// datastore, and the orchestrator's inbox is the durable guard. What this
	// buys is the common case — a redelivery inside the ack window — where a
	// second run would cost a real model call for an answer already sent.
	pub := &fakePublisher{}
	runs := 0
	counting := func(ctx context.Context, r Request) (string, error) {
		runs++
		return `{"comment":"ok"}`, nil
	}
	h := handler(counting, pub)

	h.Handle(context.Background(), req(t))
	second := req(t)
	h.Handle(context.Background(), second)

	if runs != 1 {
		t.Errorf("ran the agent %d times for one dispatch", runs)
	}
	if len(pub.published) != 2 {
		t.Errorf("published %d completions, want 2 (the answer is re-sent, not re-computed)", len(pub.published))
	}
	if !second.acked {
		t.Error("a deduped redelivery must ack")
	}
}

func TestAnUndecodableRequestNacksWithoutRunningTheAgent(t *testing.T) {
	pub := &fakePublisher{}
	runs := 0
	msg := &fakeMsg{data: []byte("not json"), attrs: map[string]string{}, attempt: 1}

	handler(func(context.Context, Request) (string, error) { runs++; return "", nil }, pub).
		Handle(context.Background(), msg)

	if runs != 0 || !msg.nacked {
		t.Errorf("runs=%d nacked=%v", runs, msg.nacked)
	}
}

func TestARequestForAnotherRoleGetsAFailedCompletionOnItsOwnLane(t *testing.T) {
	// Both OE binaries ship in ONE image. A misconfigured subscription pointing
	// the moderator at the evaluator's topic would otherwise grade with the
	// wrong agent and the output would look plausible. ADR-254 D6 (arm a): the
	// request is readable, so the parked run is told FAILED on the lane it
	// parked on (the request's reply topic) instead of waiting five deliveries
	// for a silent dead-letter.
	pub := &fakePublisher{}
	msg := req(t)
	h := NewHandler(HandlerConfig{
		AgentRole: "oe_moderate", Run: okRun("{}"), Publisher: pub,
		MaxDeliveryAttempts: 5,
	})

	h.Handle(context.Background(), msg)

	if len(pub.published) != 1 {
		t.Fatalf("published %d completions, want exactly one FAILED", len(pub.published))
	}
	got := pub.published[0]
	if got.topic != CompletionTopic("oe_evaluate") {
		t.Errorf("FAILED went to %q, want the request's lane %q", got.topic, CompletionTopic("oe_evaluate"))
	}
	if got.body.Status != StatusFailed || got.body.AgentRole != "oe_evaluate" ||
		got.body.ExecutionID != "sub-1:tsq-1:1" || got.body.ThreadID != "sub-1" {
		t.Errorf("FAILED completion = %+v, want status FAILED addressed as the request's role/thread", got.body)
	}
	if got.body.ErrorMessage == "" || !containsAll(got.body.ErrorMessage, "role_mismatch", "oe_moderate", "oe_evaluate") {
		t.Errorf("error_message %q must say role_mismatch and name both roles", got.body.ErrorMessage)
	}
	if !msg.acked || msg.nacked {
		t.Errorf("acked=%v nacked=%v, want acked once the FAILED is on the wire", msg.acked, msg.nacked)
	}
}

func TestAPartiallyReadableRequestGetsAFailedCompletion(t *testing.T) {
	// The typed decode fails (input_payload is an object, not a string) but
	// the routing fields are readable: tell the parked run FAILED rather than
	// NACK it to the DLQ in silence.
	pub := &fakePublisher{}
	raw := []byte(`{"agent_role":"oe_evaluate","execution_id":"sub-9:q-1:1","thread_id":"sub-9",` +
		`"idempotency_key":"agent_dispatch.oe_evaluate.sub-9:q-1:1","tenant_id":"t1",` +
		`"reply_topic":"` + CompletionTopic("oe_evaluate") + `","input_payload":{"mode":"evaluate"}}`)
	msg := &fakeMsg{data: raw, attrs: map[string]string{}, attempt: 1}
	runs := 0

	handler(func(context.Context, Request) (string, error) { runs++; return "", nil }, pub).
		Handle(context.Background(), msg)

	if runs != 0 {
		t.Errorf("the agent ran %d times on an undecodable request", runs)
	}
	if len(pub.published) != 1 || pub.published[0].body.Status != StatusFailed {
		t.Fatalf("want one FAILED completion, got %+v", pub.published)
	}
	got := pub.published[0]
	if got.topic != CompletionTopic("oe_evaluate") || got.body.ThreadID != "sub-9" || got.body.ExecutionID != "sub-9:q-1:1" {
		t.Errorf("FAILED completion mis-addressed: topic=%q body=%+v", got.topic, got.body)
	}
	if !containsAll(got.body.ErrorMessage, "decode_failed") {
		t.Errorf("error_message %q must say decode_failed", got.body.ErrorMessage)
	}
	if got.attrs["idempotency_key"] != "agent_dispatch.oe_evaluate.sub-9:q-1:1.completed" {
		t.Errorf("completion key attr = %q, want the request key + .completed", got.attrs["idempotency_key"])
	}
	if !msg.acked || msg.nacked {
		t.Errorf("acked=%v nacked=%v, want acked", msg.acked, msg.nacked)
	}
}

func containsAll(s string, subs ...string) bool {
	for _, sub := range subs {
		if !strings.Contains(s, sub) {
			return false
		}
	}
	return true
}

func TestAFailedPublishNacksSoTheWorkIsNotLost(t *testing.T) {
	// Acking after a failed publish would strand the parked run: the work is
	// done, nobody is ever told.
	pub := &fakePublisher{err: errors.New("topic gone")}
	msg := req(t)

	handler(okRun("{}"), pub).Handle(context.Background(), msg)

	if msg.acked || !msg.nacked {
		t.Errorf("ack=%v nack=%v, want nack", msg.acked, msg.nacked)
	}
}

func TestAReplyTopicOnTheRequestOverridesTheDefault(t *testing.T) {
	// The producer names the reply topic on the request, so a lane can be
	// repointed without redeploying the agent.
	pub := &fakePublisher{}
	var r Request
	msg := req(t)
	_ = json.Unmarshal(msg.data, &r)
	r.ReplyTopic = "chora.ai_kernel.agent_dispatch.oe_evaluate_completed.v2"
	msg.data, _ = json.Marshal(r)

	handler(okRun("{}"), pub).Handle(context.Background(), msg)

	if pub.published[0].topic != r.ReplyTopic {
		t.Errorf("topic = %q, want the request's reply_topic %q",
			pub.published[0].topic, r.ReplyTopic)
	}
}
