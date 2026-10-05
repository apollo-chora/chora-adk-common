package instancedispatch

import (
	"context"
	"errors"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

// fakeReadonlyState + other fakes shared with plugin_test.go live in fakes_test.go.

// staticResolver returns the same InstanceConfig regardless of input. Useful
// for asserting "resolver was called" without testing the resolver itself.
func staticResolver(out InstanceConfig, err error) Resolver {
	return func(ctx context.Context, id string) (InstanceConfig, error) {
		return out, err
	}
}

// captureResolver records the instanceID it was called with, returns canned data.
func captureResolver(seen *string, out InstanceConfig) Resolver {
	return func(ctx context.Context, id string) (InstanceConfig, error) {
		*seen = id
		return out, nil
	}
}

// ---------- ResolveInstanceFromState ----------

func TestResolveInstanceFromState_returnsErrorWhenStateMissingKey(t *testing.T) {
	state := &fakeReadonlyState{m: map[string]any{}}
	resolver := staticResolver(InstanceConfig{}, nil)

	_, err := ResolveInstanceFromState(context.Background(), state, "familiar_id", resolver)
	if err == nil {
		t.Fatal("want error when state missing familiar_id; got nil")
	}
	if !strings.Contains(err.Error(), "familiar_id") {
		t.Errorf("error should mention the missing state key; got %v", err)
	}
}

func TestResolveInstanceFromState_returnsErrorWhenStateValueEmpty(t *testing.T) {
	state := &fakeReadonlyState{m: map[string]any{"familiar_id": ""}}
	resolver := staticResolver(InstanceConfig{}, nil)

	_, err := ResolveInstanceFromState(context.Background(), state, "familiar_id", resolver)
	if err == nil {
		t.Fatal("want error when familiar_id is empty string; got nil")
	}
}

func TestResolveInstanceFromState_returnsErrorWhenStateValueNotString(t *testing.T) {
	state := &fakeReadonlyState{m: map[string]any{"familiar_id": 42}}
	resolver := staticResolver(InstanceConfig{}, nil)

	_, err := ResolveInstanceFromState(context.Background(), state, "familiar_id", resolver)
	if err == nil {
		t.Fatal("want error when familiar_id is not a string; got nil")
	}
}

func TestResolveInstanceFromState_callsResolverWithInstanceID(t *testing.T) {
	state := &fakeReadonlyState{m: map[string]any{"familiar_id": "math-newton-uuid"}}
	var seen string
	resolver := captureResolver(&seen, InstanceConfig{Instruction: "math prompt"})

	out, err := ResolveInstanceFromState(context.Background(), state, "familiar_id", resolver)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if seen != "math-newton-uuid" {
		t.Errorf("resolver should be called with familiar_id; got %q", seen)
	}
	if out.Instruction != "math prompt" {
		t.Errorf("returned instance should match resolver output; got %q", out.Instruction)
	}
}

func TestResolveInstanceFromState_propagatesResolverError(t *testing.T) {
	state := &fakeReadonlyState{m: map[string]any{"familiar_id": "unknown-uuid"}}
	wantErr := errors.New("not in registry")
	resolver := staticResolver(InstanceConfig{}, wantErr)

	_, err := ResolveInstanceFromState(context.Background(), state, "familiar_id", resolver)
	if !errors.Is(err, wantErr) {
		t.Errorf("want wrapped resolver error; got %v", err)
	}
}

func TestResolveInstanceFromState_isGenericOverStateKey(t *testing.T) {
	// Same plugin shape serves Course Planner (state_key="tenant_id") and
	// A2A External Mediator (state_key="familiar_exposure_grant_id").
	state := &fakeReadonlyState{m: map[string]any{"tenant_id": "acme-tenant-uuid"}}
	var seen string
	resolver := captureResolver(&seen, InstanceConfig{Instruction: "acme tenant rules"})

	_, err := ResolveInstanceFromState(context.Background(), state, "tenant_id", resolver)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if seen != "acme-tenant-uuid" {
		t.Errorf("resolver should be called with tenant_id; got %q", seen)
	}
}

func TestResolveInstanceFromState_returnsErrorWhenResolverNil(t *testing.T) {
	state := &fakeReadonlyState{m: map[string]any{"familiar_id": "x"}}
	_, err := ResolveInstanceFromState(context.Background(), state, "familiar_id", nil)
	if err == nil {
		t.Fatal("want error when resolver is nil; got nil")
	}
}

func TestResolveInstanceFromState_returnsErrorWhenStateKeyEmpty(t *testing.T) {
	state := &fakeReadonlyState{m: map[string]any{}}
	resolver := staticResolver(InstanceConfig{}, nil)
	_, err := ResolveInstanceFromState(context.Background(), state, "", resolver)
	if err == nil {
		t.Fatal("want error when stateKey is empty; got nil")
	}
}

// ---------- ApplyToolFilter ----------

func TestApplyToolFilter_dropsDisallowedFunctionDeclarations(t *testing.T) {
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{Name: "atom.search"},
						{Name: "math.solver"},
						{Name: "persona.voice"},
					},
				},
			},
		},
		Tools: map[string]any{
			"atom.search":   "impl-atom",
			"math.solver":   "impl-math",
			"persona.voice": "impl-persona",
		},
	}

	ApplyToolFilter(req, []string{"atom.search", "persona.voice"})

	if len(req.Config.Tools[0].FunctionDeclarations) != 2 {
		t.Fatalf("want 2 function declarations after filter; got %d", len(req.Config.Tools[0].FunctionDeclarations))
	}
	names := map[string]bool{}
	for _, d := range req.Config.Tools[0].FunctionDeclarations {
		names[d.Name] = true
	}
	if !names["atom.search"] || !names["persona.voice"] {
		t.Errorf("want atom.search + persona.voice retained; got %v", names)
	}
	if names["math.solver"] {
		t.Errorf("math.solver must be dropped; got %v", names)
	}
}

func TestApplyToolFilter_dropsDisallowedToolImpls(t *testing.T) {
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{},
		Tools: map[string]any{
			"atom.search":   "impl-atom",
			"math.solver":   "impl-math",
			"persona.voice": "impl-persona",
		},
	}

	ApplyToolFilter(req, []string{"atom.search"})

	if len(req.Tools) != 1 {
		t.Fatalf("want 1 tool impl after filter; got %d (have %v)", len(req.Tools), req.Tools)
	}
	if _, ok := req.Tools["atom.search"]; !ok {
		t.Error("atom.search impl must survive filter")
	}
	if _, ok := req.Tools["math.solver"]; ok {
		t.Error("math.solver impl must be dropped")
	}
}

func TestApplyToolFilter_emptyAllowedDropsAllTools(t *testing.T) {
	// Apprentice tier with zero allowed tools — model should see no tools at all.
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{
				{FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "atom.search"}}},
			},
		},
		Tools: map[string]any{"atom.search": "impl"},
	}

	ApplyToolFilter(req, []string{})

	if len(req.Tools) != 0 {
		t.Errorf("want zero tool impls; got %d", len(req.Tools))
	}
	// bug-4: the emptied tool group must be DROPPED, not left as an empty Tool —
	// gemini rejects an empty Tool (tool_type one_of uninitialized, 400). An
	// empty allow-list therefore yields zero Tools, not one empty group.
	if len(req.Config.Tools) != 0 {
		t.Errorf("want zero tool groups (empty Tool is an invalid GenerateContentRequest); got %d", len(req.Config.Tools))
	}
}

func TestApplyToolFilter_noopWhenConfigTools_Nil(t *testing.T) {
	req := &model.LLMRequest{Config: &genai.GenerateContentConfig{}, Tools: map[string]any{"a": 1}}
	ApplyToolFilter(req, []string{"a"})
	// No panic — that's the assertion.
	if _, ok := req.Tools["a"]; !ok {
		t.Error("allowed tool must survive even when Config.Tools nil")
	}
}

func TestApplyToolFilter_noopWhenRequestNil(t *testing.T) {
	// Defensive — never panic on nil. Callers should never pass nil but we
	// guard anyway because the plugin runs on every model call.
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("ApplyToolFilter(nil) should be safe; panicked: %v", r)
		}
	}()
	ApplyToolFilter(nil, []string{"x"})
}

func TestApplyToolFilter_handlesMultipleToolGroups(t *testing.T) {
	// Tools can be grouped (e.g., one group per toolset). Filter must apply
	// to declarations across ALL groups.
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{
				{FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "atom.search"}, {Name: "math.solver"}}},
				{FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "persona.voice"}}},
			},
		},
		Tools: map[string]any{
			"atom.search":   "i1",
			"math.solver":   "i2",
			"persona.voice": "i3",
		},
	}

	ApplyToolFilter(req, []string{"atom.search", "persona.voice"})

	// math.solver should be gone from group 0.
	if len(req.Config.Tools[0].FunctionDeclarations) != 1 {
		t.Errorf("group 0 should have 1 declaration after filter; got %d", len(req.Config.Tools[0].FunctionDeclarations))
	}
	// persona.voice survives in group 1.
	if len(req.Config.Tools[1].FunctionDeclarations) != 1 {
		t.Errorf("group 1 should retain persona.voice; got %d", len(req.Config.Tools[1].FunctionDeclarations))
	}
}

func TestApplyToolFilter_dropsEmptiedGroupKeepsSurvivor(t *testing.T) {
	// bug-4: when filtering empties one group but another survives, the emptied
	// group must be removed (gemini rejects empty Tools) and the survivor kept —
	// so the result has exactly one Tool, carrying persona.voice.
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{
				{FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "math.solver"}}},   // all disallowed → emptied → dropped
				{FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "persona.voice"}}}, // allowed → kept
			},
		},
		Tools: map[string]any{"math.solver": "i1", "persona.voice": "i2"},
	}

	ApplyToolFilter(req, []string{"persona.voice"})

	if len(req.Config.Tools) != 1 {
		t.Fatalf("want exactly 1 surviving tool group; got %d", len(req.Config.Tools))
	}
	if got := len(req.Config.Tools[0].FunctionDeclarations); got != 1 {
		t.Fatalf("survivor group should hold 1 declaration; got %d", got)
	}
	if name := req.Config.Tools[0].FunctionDeclarations[0].Name; name != "persona.voice" {
		t.Errorf("survivor should be persona.voice; got %q", name)
	}
}

// ---------- Config validation ----------

func TestConfig_validate_rejectsNilResolver(t *testing.T) {
	cfg := Config{StateKey: "familiar_id"}
	if err := cfg.validate(); err == nil {
		t.Fatal("want error for nil Resolver; got nil")
	}
}

func TestConfig_validate_rejectsEmptyStateKey(t *testing.T) {
	cfg := Config{Resolver: staticResolver(InstanceConfig{}, nil)}
	if err := cfg.validate(); err == nil {
		t.Fatal("want error for empty StateKey; got nil")
	}
}

func TestConfig_validate_passesWithFullConfig(t *testing.T) {
	cfg := Config{
		Resolver: staticResolver(InstanceConfig{}, nil),
		StateKey: "familiar_id",
	}
	if err := cfg.validate(); err != nil {
		t.Errorf("unexpected: %v", err)
	}
}
