package instancedispatch

import (
	"context"
	"testing"
)

func TestRequestIdentityCtxRoundTrip(t *testing.T) {
	ctx := WithRequestIdentity(context.Background(), "t1", "g1")
	if got := TenantIDFromContext(ctx); got != "t1" {
		t.Errorf("tenant: got %q want t1", got)
	}
	if got := UserGCIDFromContext(ctx); got != "g1" {
		t.Errorf("gcid: got %q want g1", got)
	}
	// Empty values are not stamped (accessors return "").
	bare := WithRequestIdentity(context.Background(), "", "")
	if TenantIDFromContext(bare) != "" || UserGCIDFromContext(bare) != "" {
		t.Error("empty identity must not stamp ctx values")
	}
}

func TestResolveInstanceFromState_StampsIdentity(t *testing.T) {
	state := &fakeReadonlyState{m: map[string]any{
		"familiar_id": "fam-1",
		"tenant_id":   "t1",
		"user_gcid":   "g1",
	}}
	var gotTenant, gotGCID string
	resolver := func(ctx context.Context, _ string) (InstanceConfig, error) {
		gotTenant = TenantIDFromContext(ctx)
		gotGCID = UserGCIDFromContext(ctx)
		return InstanceConfig{Instruction: "ok"}, nil
	}
	if _, err := ResolveInstanceFromState(context.Background(), state, "familiar_id", resolver); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotTenant != "t1" || gotGCID != "g1" {
		t.Errorf("resolver ctx missing stamped identity: tenant=%q gcid=%q", gotTenant, gotGCID)
	}
}

func TestResolveInstanceFromState_NoIdentityKeysStillResolves(t *testing.T) {
	// Crews that don't write tenant_id/user_gcid (e.g. Course Planner) still
	// resolve — the stamp is best-effort, absent keys leave ctx accessors "".
	state := &fakeReadonlyState{m: map[string]any{"familiar_id": "fam-1"}}
	var gotTenant string
	resolver := func(ctx context.Context, _ string) (InstanceConfig, error) {
		gotTenant = TenantIDFromContext(ctx)
		return InstanceConfig{}, nil
	}
	if _, err := ResolveInstanceFromState(context.Background(), state, "familiar_id", resolver); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotTenant != "" {
		t.Errorf("no tenant key → empty, got %q", gotTenant)
	}
}
