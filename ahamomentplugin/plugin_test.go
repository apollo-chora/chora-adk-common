package ahamomentplugin

import (
	"context"
	"iter"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// ahamomentplugin reads `growth_stage` AND `aha_moment_active_until` from
// session.State. If stage=3 AND now < aha_moment_active_until, OVERRIDE the
// state keys set by growthstageplugin to Stage-6 caps and set `growth.aha_active`
// = true. After the window, this plugin is a no-op.

// ---------- New() validation ----------

func TestNew_rejectsEmptyCrewKind(t *testing.T) {
	_, err := New(Config{})
	if err == nil {
		t.Fatal("want error for empty CrewKind; got nil")
	}
	if !strings.Contains(err.Error(), "CrewKind") {
		t.Errorf("error should mention CrewKind; got %v", err)
	}
}

func TestNew_buildsPluginWithExpectedName(t *testing.T) {
	p, err := New(Config{CrewKind: "familiar"})
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if !strings.Contains(p.Name(), "aha_moment") {
		t.Errorf("plugin name should describe purpose; got %q", p.Name())
	}
	if !strings.Contains(p.Name(), "familiar") {
		t.Errorf("plugin name should embed CrewKind; got %q", p.Name())
	}
}

// ---------- ApplyAhaOverride (pure function) ----------

func TestApplyAhaOverride_windowActiveStage3OverridesToStage6(t *testing.T) {
	now := time.Now()
	future := now.Add(1 * time.Hour)

	state := &fakeState{m: map[string]any{
		"growth_stage":            3,
		"aha_moment_active_until": future,
		"growth.allowed_tools":    []string{"cite_atom", "atom_search", "ebbinghaus_state"},
		"growth.memory_mode":      "rolling-7",
	}}

	active, err := ApplyAhaOverride(state, now)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if !active {
		t.Fatal("window active should return active=true")
	}

	// Verify the override propagated.
	tools, _ := state.Get("growth.allowed_tools")
	ts, ok := tools.([]string)
	if !ok {
		t.Fatalf("allowed_tools should remain []string; got %T", tools)
	}
	// ADR-249 A1a: the override advertises only SHIPPED stage-6 ladder
	// names. The raw st6 ladder still plateaus at the st4 set (ADR-218 D1),
	// but persona_lookup + score_atom_for_learner are declared unshipped,
	// so the preview's runtime lift is memory reach, not tools.
	wantTools := []string{"cite_atom"}
	for _, w := range wantTools {
		found := false
		for _, t := range ts {
			if t == w {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("after override, want %q in allowed_tools; got %v", w, ts)
		}
	}
	for _, retired := range []string{"propose_kg_merge", "suggest_atom_authoring", "query_kg"} {
		for _, got := range ts {
			if got == retired {
				t.Errorf("aha preview must not surface RETIRED innate tool %q; got %v", retired, ts)
			}
		}
	}
	// ADR-249 A1a: the preview must not advertise unshipped ladder names.
	for _, unshipped := range []string{"atom_search", "ebbinghaus_state", "persona_lookup", "score_atom_for_learner"} {
		for _, got := range ts {
			if got == unshipped {
				t.Errorf("aha preview must not surface UNSHIPPED ladder name %q; got %v", unshipped, ts)
			}
		}
	}

	mode, _ := state.Get("growth.memory_mode")
	if mode != "vector-rag-all" {
		t.Errorf("after override, want memory_mode=vector-rag-all; got %v", mode)
	}

	aha, _ := state.Get("growth.aha_active")
	if aha != true {
		t.Errorf("after override, want growth.aha_active=true; got %v", aha)
	}
}

func TestApplyAhaOverride_windowExpiredIsNoOp(t *testing.T) {
	now := time.Now()
	past := now.Add(-1 * time.Hour)

	originalTools := []string{"cite_atom", "atom_search", "ebbinghaus_state"}
	originalMode := "rolling-7"

	state := &fakeState{m: map[string]any{
		"growth_stage":            3,
		"aha_moment_active_until": past,
		"growth.allowed_tools":    originalTools,
		"growth.memory_mode":      originalMode,
	}}

	active, err := ApplyAhaOverride(state, now)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if active {
		t.Error("expired window must NOT be active")
	}

	tools, _ := state.Get("growth.allowed_tools")
	if ts, ok := tools.([]string); !ok || len(ts) != len(originalTools) {
		t.Errorf("expired window must NOT touch allowed_tools; got %v", tools)
	}
	mode, _ := state.Get("growth.memory_mode")
	if mode != originalMode {
		t.Errorf("expired window must NOT touch memory_mode; got %v", mode)
	}

	aha, _ := state.Get("growth.aha_active")
	if aha == true {
		t.Error("expired window must NOT set growth.aha_active=true")
	}
}

func TestApplyAhaOverride_wrongStageIsNoOp(t *testing.T) {
	// Defensive: window-active but stage != 3 must be a no-op. The Aha-moment
	// is exclusively a Stage-3 mechanic per ADR-149.
	now := time.Now()
	future := now.Add(1 * time.Hour)

	for _, stage := range []int{0, 1, 2, 4, 5, 6} {
		state := &fakeState{m: map[string]any{
			"growth_stage":            stage,
			"aha_moment_active_until": future,
			"growth.allowed_tools":    []string{"cite_atom"},
			"growth.memory_mode":      "rolling-1",
		}}
		active, err := ApplyAhaOverride(state, now)
		if err != nil {
			t.Fatalf("stage %d: %v", stage, err)
		}
		if active {
			t.Errorf("stage %d window-active should NOT activate aha (only stage 3 qualifies)", stage)
		}
		aha, _ := state.Get("growth.aha_active")
		if aha == true {
			t.Errorf("stage %d should not set aha_active=true; got %v", stage, aha)
		}
	}
}

func TestApplyAhaOverride_missingWindowIsNoOp(t *testing.T) {
	now := time.Now()
	state := &fakeState{m: map[string]any{
		"growth_stage":         3,
		"growth.allowed_tools": []string{"cite_atom"},
		"growth.memory_mode":   "rolling-7",
	}}
	active, err := ApplyAhaOverride(state, now)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if active {
		t.Error("missing aha_moment_active_until must not activate")
	}
}

func TestApplyAhaOverride_acceptsRFC3339String(t *testing.T) {
	// Common JSON unmarshaling path: time arrives as RFC3339 string.
	now := time.Now()
	future := now.Add(1 * time.Hour).UTC().Format(time.RFC3339)

	state := &fakeState{m: map[string]any{
		"growth_stage":            3,
		"aha_moment_active_until": future,
		"growth.allowed_tools":    []string{"cite_atom"},
		"growth.memory_mode":      "rolling-7",
	}}
	active, err := ApplyAhaOverride(state, now)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if !active {
		t.Errorf("RFC3339 string in future should activate; got active=%v", active)
	}
}

// ---------- BeforeRunCallback integration ----------

func TestBeforeRunCallback_overrideWhenWindowActive(t *testing.T) {
	p, _ := New(Config{CrewKind: "familiar"})

	state := &fakeState{m: map[string]any{
		"growth_stage":            3,
		"aha_moment_active_until": time.Now().Add(1 * time.Hour),
		"growth.allowed_tools":    []string{"cite_atom", "atom_search", "ebbinghaus_state"},
		"growth.memory_mode":      "rolling-7",
	}}
	ic := newFakeInvocationCtx(state)

	if _, err := p.BeforeRunCallback()(ic); err != nil {
		t.Fatalf("BeforeRunCallback err: %v", err)
	}

	aha, _ := state.Get("growth.aha_active")
	if aha != true {
		t.Errorf("BeforeRunCallback must set growth.aha_active when window active; got %v", aha)
	}

	mode, _ := state.Get("growth.memory_mode")
	if mode != "vector-rag-all" {
		t.Errorf("override should bump memory_mode to vector-rag-all; got %v", mode)
	}
}

func TestBeforeRunCallback_noOpWhenWindowExpired(t *testing.T) {
	p, _ := New(Config{CrewKind: "familiar"})

	state := &fakeState{m: map[string]any{
		"growth_stage":            3,
		"aha_moment_active_until": time.Now().Add(-1 * time.Hour),
		"growth.allowed_tools":    []string{"cite_atom"},
		"growth.memory_mode":      "rolling-7",
	}}
	ic := newFakeInvocationCtx(state)

	if _, err := p.BeforeRunCallback()(ic); err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	mode, _ := state.Get("growth.memory_mode")
	if mode != "rolling-7" {
		t.Errorf("expired window must not override memory_mode; got %v", mode)
	}
}

func TestApplyAhaOverride_missingStageIsNoOp(t *testing.T) {
	now := time.Now()
	state := &fakeState{m: map[string]any{
		"aha_moment_active_until": now.Add(1 * time.Hour),
	}}
	active, err := ApplyAhaOverride(state, now)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if active {
		t.Error("missing growth_stage should not activate")
	}
}

func TestApplyAhaOverride_invalidStageTypeIsNoOp(t *testing.T) {
	now := time.Now()
	state := &fakeState{m: map[string]any{
		"growth_stage":            "three",
		"aha_moment_active_until": now.Add(1 * time.Hour),
	}}
	active, err := ApplyAhaOverride(state, now)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if active {
		t.Error("non-numeric growth_stage should not activate")
	}
}

func TestApplyAhaOverride_acceptsInt64AndFloat64Stage(t *testing.T) {
	now := time.Now()
	for _, in := range []any{int(3), int32(3), int64(3), float64(3)} {
		state := &fakeState{m: map[string]any{
			"growth_stage":            in,
			"aha_moment_active_until": now.Add(1 * time.Hour),
			"growth.allowed_tools":    []string{"cite_atom"},
			"growth.memory_mode":      "rolling-7",
		}}
		active, err := ApplyAhaOverride(state, now)
		if err != nil {
			t.Fatalf("type %T: %v", in, err)
		}
		if !active {
			t.Errorf("type %T: expected active=true", in)
		}
	}
}

func TestApplyAhaOverride_acceptsTimePointer(t *testing.T) {
	now := time.Now()
	future := now.Add(1 * time.Hour)
	state := &fakeState{m: map[string]any{
		"growth_stage":            3,
		"aha_moment_active_until": &future,
		"growth.allowed_tools":    []string{"cite_atom"},
		"growth.memory_mode":      "rolling-7",
	}}
	active, err := ApplyAhaOverride(state, now)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if !active {
		t.Error("time pointer in future should activate")
	}
}

func TestApplyAhaOverride_nilTimePointerIsNoOp(t *testing.T) {
	now := time.Now()
	var nilTime *time.Time
	state := &fakeState{m: map[string]any{
		"growth_stage":            3,
		"aha_moment_active_until": nilTime,
		"growth.allowed_tools":    []string{"cite_atom"},
		"growth.memory_mode":      "rolling-7",
	}}
	active, err := ApplyAhaOverride(state, now)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if active {
		t.Error("nil time pointer should not activate")
	}
}

func TestApplyAhaOverride_invalidStringTimestampIsNoOp(t *testing.T) {
	now := time.Now()
	state := &fakeState{m: map[string]any{
		"growth_stage":            3,
		"aha_moment_active_until": "not-a-timestamp",
		"growth.allowed_tools":    []string{"cite_atom"},
		"growth.memory_mode":      "rolling-7",
	}}
	active, err := ApplyAhaOverride(state, now)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if active {
		t.Error("invalid RFC3339 string should not activate")
	}
}

func TestApplyAhaOverride_unknownTimeTypeIsNoOp(t *testing.T) {
	now := time.Now()
	state := &fakeState{m: map[string]any{
		"growth_stage":            3,
		"aha_moment_active_until": 42, // bogus int — neither time.Time nor RFC3339
		"growth.allowed_tools":    []string{"cite_atom"},
		"growth.memory_mode":      "rolling-7",
	}}
	active, err := ApplyAhaOverride(state, now)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if active {
		t.Error("non-time type should not activate")
	}
}

func TestApplyAhaOverride_rfc3339NanoString(t *testing.T) {
	now := time.Now()
	nanoFuture := now.Add(1 * time.Hour).UTC().Format(time.RFC3339Nano)
	state := &fakeState{m: map[string]any{
		"growth_stage":            3,
		"aha_moment_active_until": nanoFuture,
		"growth.allowed_tools":    []string{"cite_atom"},
		"growth.memory_mode":      "rolling-7",
	}}
	active, err := ApplyAhaOverride(state, now)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if !active {
		t.Error("RFC3339Nano string should be parseable")
	}
}

// ---------- fakes ----------

type fakeState struct{ m map[string]any }

func (f *fakeState) Get(k string) (any, error) {
	if v, ok := f.m[k]; ok {
		return v, nil
	}
	return nil, session.ErrStateKeyNotExist
}

func (f *fakeState) Set(k string, v any) error {
	f.m[k] = v
	return nil
}

func (f *fakeState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range f.m {
			if !yield(k, v) {
				return
			}
		}
	}
}

type fakeSession struct{ state *fakeState }

func (f *fakeSession) ID() string                { return "fake" }
func (f *fakeSession) AppName() string           { return "fake" }
func (f *fakeSession) UserID() string            { return "fake" }
func (f *fakeSession) State() session.State      { return f.state }
func (f *fakeSession) Events() session.Events    { return nil }
func (f *fakeSession) LastUpdateTime() time.Time { return time.Time{} }

type fakeInvocationCtx struct {
	ctx     context.Context
	session *fakeSession
}

func newFakeInvocationCtx(state *fakeState) *fakeInvocationCtx {
	return &fakeInvocationCtx{
		ctx:     context.Background(),
		session: &fakeSession{state: state},
	}
}

func (f *fakeInvocationCtx) Deadline() (time.Time, bool) { return f.ctx.Deadline() }
func (f *fakeInvocationCtx) Done() <-chan struct{}       { return f.ctx.Done() }
func (f *fakeInvocationCtx) Err() error                  { return f.ctx.Err() }
func (f *fakeInvocationCtx) Value(key any) any           { return f.ctx.Value(key) }

func (f *fakeInvocationCtx) Agent() agent.Agent          { return nil }
func (f *fakeInvocationCtx) Artifacts() agent.Artifacts  { return nil }
func (f *fakeInvocationCtx) Memory() agent.Memory        { return nil }
func (f *fakeInvocationCtx) Session() session.Session    { return f.session }
func (f *fakeInvocationCtx) InvocationID() string        { return "fake-inv" }
func (f *fakeInvocationCtx) Branch() string              { return "" }
func (f *fakeInvocationCtx) UserContent() *genai.Content { return nil }
func (f *fakeInvocationCtx) RunConfig() *agent.RunConfig { return nil }
func (f *fakeInvocationCtx) EndInvocation()              {}
func (f *fakeInvocationCtx) Ended() bool                 { return false }
func (f *fakeInvocationCtx) WithContext(ctx context.Context) agent.InvocationContext {
	return &fakeInvocationCtx{ctx: ctx, session: f.session}
}
