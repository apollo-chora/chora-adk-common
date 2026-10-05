package promptstamping

// RED-first tests for the ADR-197 M-A shared prompt-stamping primitive.
// The primitive computes a deterministic content hash over the rendered system
// prompt, captures the composition conditions, and stamps both (plus the
// prompt_version) onto the active Cloud Trace span — WITHOUT altering the
// prompt (behaviour-neutral guarantee).

import (
	"context"
	"errors"
	"iter"
	"testing"
	"time"

	otelattr "go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// ---------- fakes (package-local; agent.ReadonlyContext + session.ReadonlyState) ----------

type fakeReadonlyState struct{ m map[string]any }

func (f *fakeReadonlyState) Get(k string) (any, error) {
	if v, ok := f.m[k]; ok {
		return v, nil
	}
	return nil, session.ErrStateKeyNotExist
}
func (f *fakeReadonlyState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range f.m {
			if !yield(k, v) {
				return
			}
		}
	}
}

type fakeReadonlyCtx struct {
	ctx   context.Context
	state map[string]any
}

func (f *fakeReadonlyCtx) Deadline() (time.Time, bool) { return f.ctx.Deadline() }
func (f *fakeReadonlyCtx) Done() <-chan struct{}       { return f.ctx.Done() }
func (f *fakeReadonlyCtx) Err() error                  { return f.ctx.Err() }
func (f *fakeReadonlyCtx) Value(key any) any           { return f.ctx.Value(key) }
func (f *fakeReadonlyCtx) UserContent() *genai.Content { return nil }
func (f *fakeReadonlyCtx) InvocationID() string        { return "fake-inv" }
func (f *fakeReadonlyCtx) AgentName() string           { return "fake-agent" }
func (f *fakeReadonlyCtx) UserID() string              { return "fake-user" }
func (f *fakeReadonlyCtx) AppName() string             { return "fake-app" }
func (f *fakeReadonlyCtx) SessionID() string           { return "fake-session" }
func (f *fakeReadonlyCtx) Branch() string              { return "" }
func (f *fakeReadonlyCtx) ReadonlyState() session.ReadonlyState {
	return &fakeReadonlyState{m: f.state}
}

func newCtx(ctx context.Context, state map[string]any) agent.ReadonlyContext {
	return &fakeReadonlyCtx{ctx: ctx, state: state}
}

// ---------- ContentHash ----------

func TestContentHash_knownVector(t *testing.T) {
	// sha256("abc") canonical vector.
	const want = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
	if got := ContentHash("abc"); got != want {
		t.Errorf("ContentHash(abc)=%q want %q", got, want)
	}
}

func TestContentHash_deterministicAndDistinct(t *testing.T) {
	if ContentHash("same") != ContentHash("same") {
		t.Error("ContentHash must be deterministic")
	}
	if ContentHash("a") == ContentHash("b") {
		t.Error("distinct prompts must hash differently")
	}
}

// ---------- Evidence.SpanAttributes ----------

func TestEvidence_SpanAttributes_versionAndHash(t *testing.T) {
	ev := Evidence{PromptVersion: "v1", ContentHash: "deadbeef"}
	attrs := ev.SpanAttributes()
	if !hasString(attrs, "chora.prompt.version", "v1") {
		t.Errorf("missing chora.prompt.version; got %v", attrs)
	}
	if !hasString(attrs, "chora.prompt.content_hash", "deadbeef") {
		t.Errorf("missing chora.prompt.content_hash; got %v", attrs)
	}
}

func TestEvidence_SpanAttributes_conditionsPrefixed(t *testing.T) {
	ev := Evidence{
		PromptVersion: "v1",
		ContentHash:   "h",
		Conditions:    map[string]string{"intent": "new_question", "question_type": "mcq"},
	}
	attrs := ev.SpanAttributes()
	if !hasString(attrs, "chora.prompt.condition.intent", "new_question") {
		t.Errorf("missing condition.intent; got %v", attrs)
	}
	if !hasString(attrs, "chora.prompt.condition.question_type", "mcq") {
		t.Errorf("missing condition.question_type; got %v", attrs)
	}
}

// ---------- WithStamping ----------

func TestWithStamping_returnsPromptUnchanged(t *testing.T) {
	inner := func(agent.ReadonlyContext) (string, error) { return "RENDERED PROMPT", nil }
	wrapped := WithStamping("v1", nil, inner)
	got, err := wrapped(newCtx(context.Background(), nil))
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if got != "RENDERED PROMPT" {
		t.Errorf("prompt must be unchanged; got %q", got)
	}
}

func TestWithStamping_propagatesInnerError(t *testing.T) {
	sentinel := errors.New("compose failed")
	inner := func(agent.ReadonlyContext) (string, error) { return "", sentinel }
	wrapped := WithStamping("v1", nil, inner)
	_, err := wrapped(newCtx(context.Background(), nil))
	if !errors.Is(err, sentinel) {
		t.Errorf("inner error must propagate; got %v", err)
	}
}

func TestWithStamping_nilExtractorNoPanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("nil extractor must be safe; panicked: %v", r)
		}
	}()
	inner := func(agent.ReadonlyContext) (string, error) { return "p", nil }
	wrapped := WithStamping("v1", nil, inner)
	if _, err := wrapped(newCtx(context.Background(), nil)); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
}

func TestWithStamping_stampsActiveSpan(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	ctx, span := tp.Tracer("test").Start(context.Background(), "compose")

	extract := func(s session.ReadonlyState) map[string]string {
		out := map[string]string{}
		if v, err := s.Get("intent"); err == nil {
			out["intent"], _ = v.(string)
		}
		return out
	}
	inner := func(agent.ReadonlyContext) (string, error) { return "the prompt", nil }
	wrapped := WithStamping("v1", extract, inner)

	if _, err := wrapped(newCtx(ctx, map[string]any{"intent": "new_question"})); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	span.End()

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("want 1 ended span; got %d", len(spans))
	}
	attrs := spans[0].Attributes()
	if !hasString(attrs, "chora.prompt.version", "v1") {
		t.Errorf("span missing prompt.version; attrs=%v", attrs)
	}
	if !hasString(attrs, "chora.prompt.content_hash", ContentHash("the prompt")) {
		t.Errorf("span missing/incorrect content_hash; attrs=%v", attrs)
	}
	if !hasString(attrs, "chora.prompt.condition.intent", "new_question") {
		t.Errorf("span missing condition.intent; attrs=%v", attrs)
	}
}

func TestWithStamping_stateResolvedVersionOverridesDefault(t *testing.T) {
	// ADR-197 M-B: when the orchestrator has resolved a version into session
	// state, the span MUST carry that version, NOT the agentconfig fallback
	// passed at wrap time.
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	ctx, span := tp.Tracer("test").Start(context.Background(), "compose")

	inner := func(agent.ReadonlyContext) (string, error) { return "the prompt", nil }
	wrapped := WithStamping("v1-fallback", nil, inner)

	state := map[string]any{StateKeyResolvedPromptVersion: "v7-resolved"}
	if _, err := wrapped(newCtx(ctx, state)); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	span.End()

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("want 1 ended span; got %d", len(spans))
	}
	attrs := spans[0].Attributes()
	if !hasString(attrs, "chora.prompt.version", "v7-resolved") {
		t.Errorf("span must carry the state-resolved version; attrs=%v", attrs)
	}
	if hasString(attrs, "chora.prompt.version", "v1-fallback") {
		t.Errorf("span must NOT carry the fallback when state resolves a version; attrs=%v", attrs)
	}
}

func TestWithStamping_absentOrEmptyResolvedVersionFallsBack(t *testing.T) {
	// Behaviour-neutral guarantee: with no resolved_prompt_version key (or an
	// empty / non-string value) the span carries the passed fallback exactly
	// as it did before ADR-197 M-B.
	cases := map[string]map[string]any{
		"absent key":             nil,
		"empty string value":     {StateKeyResolvedPromptVersion: ""},
		"non-string value":       {StateKeyResolvedPromptVersion: 42},
		"unrelated keys present": {"intent": "new_question"},
	}
	for name, state := range cases {
		t.Run(name, func(t *testing.T) {
			recorder := tracetest.NewSpanRecorder()
			tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
			ctx, span := tp.Tracer("test").Start(context.Background(), "compose")

			inner := func(agent.ReadonlyContext) (string, error) { return "p", nil }
			wrapped := WithStamping("v1", nil, inner)
			if _, err := wrapped(newCtx(ctx, state)); err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			span.End()

			attrs := recorder.Ended()[0].Attributes()
			if !hasString(attrs, "chora.prompt.version", "v1") {
				t.Errorf("span must fall back to the passed version; attrs=%v", attrs)
			}
		})
	}
}

func TestResolvedVersion_nilStateFallsBack(t *testing.T) {
	// Defensive: a nil ReadonlyState must yield the fallback, never panic.
	if got := resolvedVersion(nil, "v1"); got != "v1" {
		t.Errorf("resolvedVersion(nil)=%q want fallback %q", got, "v1")
	}
}

func TestStamp_noopWithoutActiveSpan(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Stamp without active span must be safe; panicked: %v", r)
		}
	}()
	Stamp(context.Background(), Evidence{PromptVersion: "v1", ContentHash: "h"})
}

// ---------- helpers ----------

func hasString(attrs []otelattr.KeyValue, key, want string) bool {
	for _, a := range attrs {
		if string(a.Key) == key && a.Value.AsString() == want {
			return true
		}
	}
	return false
}
