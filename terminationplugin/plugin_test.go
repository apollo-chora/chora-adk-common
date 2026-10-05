// Tests for terminationplugin — W3 foundation Phase 4 (2026-05-12).

package terminationplugin

import (
	"context"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"testing"

	"google.golang.org/adk/agent"
)

// recordingPublisher captures emitted events; the optional emitErr forces
// Publish to return an error, exercising the swallow-and-log path.
type recordingPublisher struct {
	mu      sync.Mutex
	events  []AgentTerminatedEvent
	emitErr error
}

func (r *recordingPublisher) Publish(ctx context.Context, e AgentTerminatedEvent) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, e)
	return r.emitErr
}

func (r *recordingPublisher) snapshot() []AgentTerminatedEvent {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]AgentTerminatedEvent, len(r.events))
	copy(out, r.events)
	return out
}

func cfg(p Publisher) Config {
	return Config{
		Publisher:   p,
		AgentID:     "familiar_companion",
		Runtime:     "AGENT_EXECUTION_RUNTIME_ADK_GO",
		CrewKind:    "familiar",
		CrewPattern: "P1_SINGLE_AGENT_REACT",
	}
}

// ----------------------------------------------------------------------------
// Config validation
// ----------------------------------------------------------------------------

func TestNew_RejectsNilPublisher(t *testing.T) {
	t.Parallel()
	_, err := New(Config{AgentID: "a", Runtime: "r"})
	if err == nil {
		t.Fatal("expected error on nil Publisher")
	}
}

func TestNew_RejectsEmptyAgentID(t *testing.T) {
	t.Parallel()
	_, err := New(Config{Publisher: &recordingPublisher{}, Runtime: "r"})
	if err == nil {
		t.Fatal("expected error on empty AgentID")
	}
}

func TestNew_RejectsEmptyRuntime(t *testing.T) {
	t.Parallel()
	_, err := New(Config{Publisher: &recordingPublisher{}, AgentID: "a"})
	if err == nil {
		t.Fatal("expected error on empty Runtime")
	}
}

func TestNew_AcceptsMinimalConfig(t *testing.T) {
	t.Parallel()
	p, err := New(cfg(&recordingPublisher{}))
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if p == nil {
		t.Fatal("plugin should not be nil")
	}
	if !strings.HasPrefix(p.Name(), "chora_termination_") {
		t.Errorf("plugin name not chora_termination_*: %s", p.Name())
	}
}

func TestNew_WiresExpectedCallbacks(t *testing.T) {
	t.Parallel()
	p, err := New(cfg(&recordingPublisher{}))
	if err != nil {
		t.Fatal(err)
	}
	if p.BeforeRunCallback() == nil {
		t.Error("BeforeRunCallback must be wired")
	}
	if p.AfterRunCallback() == nil {
		t.Error("AfterRunCallback must be wired")
	}
	if p.OnModelErrorCallback() == nil {
		t.Error("OnModelErrorCallback must be wired (captures last LLM error)")
	}
	if p.OnToolErrorCallback() == nil {
		t.Error("OnToolErrorCallback must be wired (captures last tool error)")
	}
}

// ----------------------------------------------------------------------------
// Success path — AfterRun emits SUCCESS when no error was captured
// ----------------------------------------------------------------------------

func TestAfterRun_EmitsSuccessWhenNoError(t *testing.T) {
	t.Parallel()
	pub := &recordingPublisher{}
	plug, err := New(cfg(pub))
	if err != nil {
		t.Fatal(err)
	}
	ic := newFakeICtx("familiar_companion", "session-1", map[string]any{
		"tenant_id": "tenant-A",
		"user_gcid": "user-1",
	})
	_, err = plug.BeforeRunCallback()(ic)
	if err != nil {
		t.Fatalf("BeforeRunCallback err: %v", err)
	}
	plug.AfterRunCallback()(ic)
	events := pub.snapshot()
	if len(events) != 1 {
		t.Fatalf("event count=%d, want 1", len(events))
	}
	e := events[0]
	if e.TerminationCode != CodeSuccess {
		t.Errorf("code=%q, want %q", e.TerminationCode, CodeSuccess)
	}
	if e.AgentID != "familiar_companion" {
		t.Errorf("agent_id=%q, want familiar_companion", e.AgentID)
	}
	if e.Runtime != "AGENT_EXECUTION_RUNTIME_ADK_GO" {
		t.Errorf("runtime=%q", e.Runtime)
	}
	if e.TenantID != "tenant-A" {
		t.Errorf("tenant_id=%q, want tenant-A", e.TenantID)
	}
	if e.GCID != "user-1" {
		t.Errorf("gcid=%q, want user-1", e.GCID)
	}
	if e.ExecutionID != "session-1" {
		t.Errorf("execution_id=%q, want session-1 (the session.ID())", e.ExecutionID)
	}
	if e.CrewPattern != "P1_SINGLE_AGENT_REACT" {
		t.Errorf("crew_pattern=%q", e.CrewPattern)
	}
	if e.CrewID != "familiar" {
		t.Errorf("crew_id=%q (using CrewKind)", e.CrewID)
	}
	if e.TerminatedAt.IsZero() {
		t.Error("terminated_at zero — must be populated")
	}
}

// ----------------------------------------------------------------------------
// Failure path — OnModelError captures, AfterRun emits RUNTIME_ERROR
// ----------------------------------------------------------------------------

func TestModelError_CapturesAndAfterRunEmitsRuntimeError(t *testing.T) {
	t.Parallel()
	pub := &recordingPublisher{}
	plug, err := New(cfg(pub))
	if err != nil {
		t.Fatal(err)
	}
	state := map[string]any{
		"tenant_id": "tenant-B",
		"user_gcid": "user-2",
	}
	_, _ = plug.BeforeRunCallback()(newFakeICtx("familiar_companion", "session-2", state))

	cb := plug.OnModelErrorCallback()
	if cb == nil {
		t.Fatal("OnModelErrorCallback should be wired")
	}
	// Drive the OnModelError path — the plugin captures the error into
	// session state so AfterRun finds it.
	cbCtx := newFakeCallbackCtx(state)
	_, retErr := cb(cbCtx, nil, errors.New("vertex 429 throttled"))
	if retErr != nil {
		t.Errorf("OnModelError should not swallow the upstream error (returned %v)", retErr)
	}
	plug.AfterRunCallback()(newFakeICtx("familiar_companion", "session-2", state))

	events := pub.snapshot()
	if len(events) != 1 {
		t.Fatalf("event count=%d, want 1", len(events))
	}
	e := events[0]
	if e.TerminationCode != CodeRuntimeError {
		t.Errorf("code=%q, want %q", e.TerminationCode, CodeRuntimeError)
	}
	if !strings.Contains(e.LastErrorMessage, "vertex 429 throttled") {
		t.Errorf("last_error_message=%q", e.LastErrorMessage)
	}
	if e.LastStateNode != "model_call" {
		t.Errorf("last_state_node=%q, want model_call", e.LastStateNode)
	}
}

func TestToolError_CapturesAndAfterRunEmitsRuntimeError(t *testing.T) {
	t.Parallel()
	pub := &recordingPublisher{}
	plug, err := New(cfg(pub))
	if err != nil {
		t.Fatal(err)
	}
	state := map[string]any{
		"tenant_id": "tenant-C",
		"user_gcid": "user-3",
	}
	_, _ = plug.BeforeRunCallback()(newFakeICtx("familiar_companion", "session-3", state))

	cb := plug.OnToolErrorCallback()
	if cb == nil {
		t.Fatal("OnToolErrorCallback should be wired")
	}
	// The plugin captures via the public CaptureToolError helper directly
	// when adk SDK's tool.Context isn't easily constructible in tests.
	// Drive the helper that the SDK callback delegates to.
	CaptureToolErrorOnState(&fakeState{m: state}, "vector_search", errors.New("DB timeout"))

	plug.AfterRunCallback()(newFakeICtx("familiar_companion", "session-3", state))

	events := pub.snapshot()
	if len(events) != 1 {
		t.Fatalf("event count=%d", len(events))
	}
	e := events[0]
	if e.TerminationCode != CodeRuntimeError {
		t.Errorf("code=%q, want RUNTIME_ERROR", e.TerminationCode)
	}
	if e.LastToolName != "vector_search" {
		t.Errorf("last_tool_name=%q, want vector_search", e.LastToolName)
	}
	if !strings.Contains(e.LastErrorMessage, "DB timeout") {
		t.Errorf("last_error_message=%q", e.LastErrorMessage)
	}
	if e.LastStateNode != "tool_call" {
		t.Errorf("last_state_node=%q, want tool_call", e.LastStateNode)
	}
}

// ----------------------------------------------------------------------------
// Missing tenant_id/user_gcid — emission STILL fires (operator needs to
// see SOMETHING in the termination stream even when state is incomplete),
// but the event carries empty attribution.
// ----------------------------------------------------------------------------

func TestAfterRun_EmitsEvenWithoutTenantOrGCID(t *testing.T) {
	t.Parallel()
	pub := &recordingPublisher{}
	plug, err := New(cfg(pub))
	if err != nil {
		t.Fatal(err)
	}
	ic := newFakeICtx("familiar_companion", "session-x", map[string]any{})
	_, _ = plug.BeforeRunCallback()(ic)
	plug.AfterRunCallback()(ic)
	events := pub.snapshot()
	if len(events) != 1 {
		t.Fatalf("expected emission even on missing attribution; got %d", len(events))
	}
	if events[0].TenantID != "" || events[0].GCID != "" {
		t.Errorf("attribution should be empty: tenant=%q gcid=%q",
			events[0].TenantID, events[0].GCID)
	}
}

// ----------------------------------------------------------------------------
// Publisher error — logged + swallowed; doesn't panic the AfterRun
// ----------------------------------------------------------------------------

func TestAfterRun_SwallowsPublisherError(t *testing.T) {
	t.Parallel()
	pub := &recordingPublisher{emitErr: errors.New("outbox unavailable")}
	plug, err := New(cfg(pub))
	if err != nil {
		t.Fatal(err)
	}
	ic := newFakeICtx("familiar_companion", "session-err", map[string]any{
		"tenant_id": "tenant-A",
		"user_gcid": "user-1",
	})
	_, _ = plug.BeforeRunCallback()(ic)
	// Must not panic.
	plug.AfterRunCallback()(ic)
	if len(pub.snapshot()) != 1 {
		t.Error("Publisher should have been called even though it returned an error")
	}
}

// ----------------------------------------------------------------------------
// BeforeRun resets per-run state (e.g., a second run on the same session
// must not inherit the prior run's error capture)
// ----------------------------------------------------------------------------

func TestBeforeRun_ResetsErrorCapture(t *testing.T) {
	t.Parallel()
	pub := &recordingPublisher{}
	plug, err := New(cfg(pub))
	if err != nil {
		t.Fatal(err)
	}
	state := map[string]any{
		"tenant_id":           "tenant-A",
		"user_gcid":           "user-1",
		stateKeyLastError:     "leftover from previous run",
		stateKeyLastErrorStep: "model_call",
	}
	ic := newFakeICtx("familiar_companion", "session-reset", state)
	_, _ = plug.BeforeRunCallback()(ic)
	plug.AfterRunCallback()(ic)
	events := pub.snapshot()
	if len(events) != 1 {
		t.Fatalf("event count=%d", len(events))
	}
	if events[0].TerminationCode != CodeSuccess {
		t.Errorf("BeforeRun must clear stale error state — got code %q",
			events[0].TerminationCode)
	}
}

// Compile-time check — the package-level Publisher type behaves as expected.
var _ Publisher = (*recordingPublisher)(nil)

// Suppress unused import on `agent` — needed when we add more direct callback
// invocations.
var _ = agent.RunConfig{}

// ----------------------------------------------------------------------------
// OnModelError callback — direct drive through the SDK signature
// ----------------------------------------------------------------------------

func TestOnModelErrorCallback_CapturesViaCallbackContext(t *testing.T) {
	t.Parallel()
	pub := &recordingPublisher{}
	plug, err := New(cfg(pub))
	if err != nil {
		t.Fatal(err)
	}

	state := map[string]any{}
	cbCtx := newFakeCallbackCtx(state)

	cb := plug.OnModelErrorCallback()
	resp, retErr := cb(cbCtx, nil, errors.New("model 500"))
	if resp != nil {
		t.Errorf("OnModelError should return nil response, got %v", resp)
	}
	if retErr != nil {
		t.Errorf("OnModelError should NOT swallow caller error, got %v", retErr)
	}
	if got := state[stateKeyLastError].(string); !strings.Contains(got, "model 500") {
		t.Errorf("last_error not captured: %q", got)
	}
	if got := state[stateKeyLastErrorStep].(string); got != "model_call" {
		t.Errorf("last_error_step=%q, want model_call", got)
	}
}

// Nil error on the model-error callback is a defensive no-op — the SDK
// would never call OnModelError with nil, but a future hook might.
func TestCaptureErrorOnState_NoOpOnNilErr(t *testing.T) {
	t.Parallel()
	state := &fakeState{m: map[string]any{}}
	captureErrorOnState(state, "model_call", "", nil)
	if _, err := state.Get(stateKeyLastError); err == nil {
		t.Error("captureErrorOnState(nil) should not write state")
	}
}

// ----------------------------------------------------------------------------
// IterationCount round-trips from state to the emitted event
// ----------------------------------------------------------------------------

func TestEmit_IterationCountRoundTrips(t *testing.T) {
	t.Parallel()
	pub := &recordingPublisher{}
	plug, err := New(cfg(pub))
	if err != nil {
		t.Fatal(err)
	}
	state := map[string]any{
		"tenant_id":            "tenant-A",
		"user_gcid":            "user-1",
		stateKeyIterationCount: int64(7),
	}
	ic := newFakeICtx("familiar_companion", "session-iter", state)
	// Note: BeforeRunCallback resets iteration_count to 0. To validate
	// round-trip the way prod runs would (iteration incremented during
	// the run), we skip BeforeRun and directly drive AfterRun here.
	plug.AfterRunCallback()(ic)
	events := pub.snapshot()
	if got := events[0].IterationCount; got != 7 {
		t.Errorf("IterationCount=%d, want 7", got)
	}
}

// ----------------------------------------------------------------------------
// getInt — exercise the float64 and missing-key branches
// ----------------------------------------------------------------------------

func TestGetInt_HandlesAllNumericTypes(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		val  any
		want int
	}{
		{"int", 3, 3},
		{"int64", int64(4), 4},
		{"float64", float64(5.7), 5},
		{"string-fallback-to-zero", "nope", 0},
		{"bool-fallback-to-zero", true, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := &fakeState{m: map[string]any{"k": tc.val}}
			if got := getInt(s, "k"); got != tc.want {
				t.Errorf("getInt(%v)=%d, want %d", tc.val, got, tc.want)
			}
		})
	}
}

func TestGetInt_MissingKeyReturnsZero(t *testing.T) {
	t.Parallel()
	s := &fakeState{m: map[string]any{}}
	if got := getInt(s, "missing"); got != 0 {
		t.Errorf("missing key getInt()=%d, want 0", got)
	}
}

func TestGetString_NonStringValueReturnsEmpty(t *testing.T) {
	t.Parallel()
	s := &fakeState{m: map[string]any{"k": 123}}
	if got := getString(s, "k"); got != "" {
		t.Errorf("non-string getString()=%q, want empty", got)
	}
}

// ----------------------------------------------------------------------------
// Long error message — emit truncates to 300 chars
// ----------------------------------------------------------------------------

func TestEmit_TruncatesLongLastError(t *testing.T) {
	t.Parallel()
	long := strings.Repeat("X", 500) + "_TAIL"
	pub := &recordingPublisher{}
	plug, err := New(cfg(pub))
	if err != nil {
		t.Fatal(err)
	}
	state := map[string]any{
		"tenant_id":       "tenant-A",
		"user_gcid":       "user-1",
		stateKeyLastError: long,
	}
	ic := newFakeICtx("familiar_companion", "session-trunc", state)
	plug.AfterRunCallback()(ic)
	events := pub.snapshot()
	if got := len(events[0].LastErrorMessage); got > 300 {
		t.Errorf("LastErrorMessage len=%d, want <=300", got)
	}
	if strings.Contains(events[0].LastErrorMessage, "_TAIL") {
		t.Error("trunc did not clip the tail")
	}
}

// ----------------------------------------------------------------------------
// WithLogger override (Config.Logger non-nil)
// ----------------------------------------------------------------------------

func TestConfigLogger_OverrideIsUsed(t *testing.T) {
	t.Parallel()
	// Smoke-check: passing a non-nil logger doesn't error. The output
	// path is exercised in TestAfterRun_SwallowsPublisherError above
	// (which logs the failure); here we just confirm the field accepts
	// a value.
	c := cfg(&recordingPublisher{})
	logger := newDiscardLogger()
	c.Logger = logger
	if _, err := New(c); err != nil {
		t.Errorf("non-nil Logger should be accepted: %v", err)
	}
}

func newDiscardLogger() *slog.Logger {
	var buf strings.Builder
	return slog.New(slog.NewTextHandler(&discardWriter{b: &buf}, &slog.HandlerOptions{Level: slog.LevelError}))
}

type discardWriter struct{ b *strings.Builder }

func (d *discardWriter) Write(p []byte) (int, error) { return len(p), nil }

// ----------------------------------------------------------------------------
// LoggingPublisher
// ----------------------------------------------------------------------------

func TestLoggingPublisher_PublishesStructuredRecord(t *testing.T) {
	t.Parallel()
	var buf strings.Builder
	pub := &LoggingPublisher{
		Logger: slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelInfo})),
	}
	err := pub.Publish(context.Background(), AgentTerminatedEvent{
		AgentID:         "qgen_p2_generator",
		ExecutionID:     "session-q1",
		Runtime:         "AGENT_EXECUTION_RUNTIME_ADK_GO",
		TerminationCode: CodeSuccess,
		CrewID:          "qgen",
		TenantID:        "tenant-A",
		GCID:            "user-1",
	})
	if err != nil {
		t.Errorf("LoggingPublisher.Publish should never return error, got %v", err)
	}
	out := buf.String()
	for _, want := range []string{
		"agent_terminated_emit",
		"qgen_p2_generator",
		"session-q1",
		CodeSuccess,
		"tenant-A",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("log output missing %q: %s", want, out)
		}
	}
}

func TestLoggingPublisher_NilLoggerUsesDefault(t *testing.T) {
	t.Parallel()
	pub := &LoggingPublisher{}
	// Smoke — must not panic, must not error.
	if err := pub.Publish(context.Background(), AgentTerminatedEvent{AgentID: "x", ExecutionID: "y", Runtime: "r", TerminationCode: CodeSuccess}); err != nil {
		t.Errorf("nil logger should default to slog.Default(), got err %v", err)
	}
}

// ----------------------------------------------------------------------------
// MaxIterations cap (Option A — counter-via-plugin)
// W3 foundation Phase 6 (2026-05-12)
// ----------------------------------------------------------------------------

func TestNew_MaxIterationsZeroMeansUnlimited(t *testing.T) {
	t.Parallel()
	c := cfg(&recordingPublisher{})
	c.MaxIterations = 0
	plug, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	// With cap unlimited, BeforeModelCallback should still be wired
	// (the counter is incremented even when no cap fires — useful for
	// the IterationCount on the emitted event).
	if plug.BeforeModelCallback() == nil {
		t.Error("BeforeModelCallback must be wired (it increments the per-run counter)")
	}
}

func TestBeforeModelCallback_IncrementsIterationCounter(t *testing.T) {
	t.Parallel()
	c := cfg(&recordingPublisher{})
	c.MaxIterations = 10
	plug, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	state := map[string]any{
		"tenant_id": "tenant-A",
		"user_gcid": "user-1",
	}
	cb := plug.BeforeModelCallback()
	ctx := newFakeCallbackCtx(state)
	for i := 0; i < 4; i++ {
		_, err := cb(ctx, nil)
		if err != nil {
			t.Fatalf("iteration %d returned err: %v", i, err)
		}
	}
	if got := getInt(&fakeState{m: state}, stateKeyIterationCount); got != 4 {
		t.Errorf("iteration count=%d, want 4", got)
	}
}

func TestBeforeModelCallback_AbortsWhenCapExceeded(t *testing.T) {
	t.Parallel()
	pub := &recordingPublisher{}
	c := cfg(pub)
	c.MaxIterations = 3
	plug, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	state := map[string]any{
		"tenant_id": "tenant-A",
		"user_gcid": "user-1",
	}
	cb := plug.BeforeModelCallback()
	ctx := newFakeCallbackCtx(state)
	// First 3 iterations should pass.
	for i := 0; i < 3; i++ {
		_, err := cb(ctx, nil)
		if err != nil {
			t.Fatalf("iteration %d should pass under cap=3, got err: %v", i, err)
		}
	}
	// 4th iteration trips the cap → returns error to abort the run.
	_, err = cb(ctx, nil)
	if err == nil {
		t.Fatal("expected cap-exceeded error on iteration 4")
	}
	if !strings.Contains(err.Error(), "max_iterations") {
		t.Errorf("error should mention max_iterations: %q", err.Error())
	}
	// The cap-tripping iteration must mark the step so AfterRun emits
	// MAX_ITERATIONS rather than RUNTIME_ERROR.
	if got := getString(&fakeState{m: state}, stateKeyLastErrorStep); got != "max_iterations" {
		t.Errorf("last_error_step=%q, want max_iterations", got)
	}
}

func TestAfterRun_EmitsMaxIterationsCodeWhenCapTripped(t *testing.T) {
	t.Parallel()
	pub := &recordingPublisher{}
	c := cfg(pub)
	c.MaxIterations = 2
	plug, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	state := map[string]any{
		"tenant_id": "tenant-A",
		"user_gcid": "user-1",
	}
	ic := newFakeICtx("familiar_companion", "session-cap", state)
	_, _ = plug.BeforeRunCallback()(ic)

	cb := plug.BeforeModelCallback()
	cbCtx := newFakeCallbackCtx(state)
	_, _ = cb(cbCtx, nil)
	_, _ = cb(cbCtx, nil)
	// 3rd trips cap=2.
	if _, err := cb(cbCtx, nil); err == nil {
		t.Fatal("expected cap-exceeded error")
	}

	plug.AfterRunCallback()(ic)
	events := pub.snapshot()
	if len(events) != 1 {
		t.Fatalf("event count=%d", len(events))
	}
	if events[0].TerminationCode != CodeMaxIterations {
		t.Errorf("termination_code=%q, want %q",
			events[0].TerminationCode, CodeMaxIterations)
	}
	if events[0].IterationCount != 3 {
		t.Errorf("iteration_count=%d, want 3 (capped at 2+1 retry attempt)",
			events[0].IterationCount)
	}
}

func TestBeforeRun_ResetsIterationCounter(t *testing.T) {
	t.Parallel()
	c := cfg(&recordingPublisher{})
	c.MaxIterations = 5
	plug, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	state := map[string]any{
		"tenant_id":            "tenant-A",
		"user_gcid":            "user-1",
		stateKeyIterationCount: 99,
	}
	ic := newFakeICtx("familiar_companion", "session-reset-iter", state)
	_, _ = plug.BeforeRunCallback()(ic)
	if got := getInt(&fakeState{m: state}, stateKeyIterationCount); got != 0 {
		t.Errorf("BeforeRun should reset iteration counter to 0, got %d", got)
	}
}

func TestAfterRun_PrefersTheDispatchExecutionIDOverTheSessionID(t *testing.T) {
	t.Parallel()
	pub := &recordingPublisher{}
	plug, err := New(cfg(pub))
	if err != nil {
		t.Fatal(err)
	}
	// A persistent conversation: the session id is the conversation, the
	// dispatch names the run (agentdispatch stamps dispatch_execution_id).
	ic := newFakeICtx("companion_chat", "1071b4f1-conversation", map[string]any{
		"tenant_id":                 "tenant-A",
		"user_gcid":                 "user-1",
		StateKeyDispatchExecutionID: "probe:typed-1",
	})
	if _, err := plug.BeforeRunCallback()(ic); err != nil {
		t.Fatalf("BeforeRunCallback err: %v", err)
	}
	plug.AfterRunCallback()(ic)
	events := pub.snapshot()
	if len(events) != 1 {
		t.Fatalf("event count=%d, want 1", len(events))
	}
	if events[0].ExecutionID != "probe:typed-1" {
		t.Errorf("execution_id=%q, want the dispatch execution id", events[0].ExecutionID)
	}
	// A non-string or empty value falls back to the session id.
	ic = newFakeICtx("companion_chat", "dispatch:e-9", map[string]any{StateKeyDispatchExecutionID: 42})
	_, _ = plug.BeforeRunCallback()(ic)
	plug.AfterRunCallback()(ic)
	events = pub.snapshot()
	if events[len(events)-1].ExecutionID != "dispatch:e-9" {
		t.Errorf("execution_id=%q, want the session id fallback", events[len(events)-1].ExecutionID)
	}
}
