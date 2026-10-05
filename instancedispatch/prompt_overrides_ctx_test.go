package instancedispatch

// ADR-197 P3 (CHO-2368): the prompt-override map resolved by the session
// creator (chora-consumption for the familiar) rides session state as
// prompt_overrides_json and must reach the instance resolver the same way
// familiar_config does - stamped onto ctx by ResolveInstanceFromState.
// Best-effort + additive: crews that never write the key are unaffected.

import (
	"context"
	"testing"
)

func TestPromptOverridesCtxRoundTrip(t *testing.T) {
	ctx := WithPromptOverridesJSON(context.Background(), `{"role_frame":"X"}`)
	if got := PromptOverridesJSONFromContext(ctx); got != `{"role_frame":"X"}` {
		t.Errorf("overrides json: got %q", got)
	}
	// Empty values are not stamped (accessor returns "").
	bare := WithPromptOverridesJSON(context.Background(), "")
	if PromptOverridesJSONFromContext(bare) != "" {
		t.Error("empty overrides json must not stamp a ctx value")
	}
}

func TestResolveInstanceFromState_StampsPromptOverrides(t *testing.T) {
	state := &fakeReadonlyState{m: map[string]any{
		"familiar_id":           "fam-1",
		"prompt_overrides_json": `{"task_frame":"OVR"}`,
	}}
	var got string
	resolver := func(ctx context.Context, _ string) (InstanceConfig, error) {
		got = PromptOverridesJSONFromContext(ctx)
		return InstanceConfig{Instruction: "ok"}, nil
	}
	if _, err := ResolveInstanceFromState(context.Background(), state, "familiar_id", resolver); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != `{"task_frame":"OVR"}` {
		t.Errorf("resolver ctx missing stamped overrides: got %q", got)
	}
}

func TestResolveInstanceFromState_NoOverridesKeyStillResolves(t *testing.T) {
	state := &fakeReadonlyState{m: map[string]any{"familiar_id": "fam-1"}}
	var got string
	resolver := func(ctx context.Context, _ string) (InstanceConfig, error) {
		got = PromptOverridesJSONFromContext(ctx)
		return InstanceConfig{}, nil
	}
	if _, err := ResolveInstanceFromState(context.Background(), state, "familiar_id", resolver); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "" {
		t.Errorf("no overrides key -> empty, got %q", got)
	}
}
