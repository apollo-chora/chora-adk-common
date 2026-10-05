// Tests for ADR-169 per-request tenant propagation — session state
// tenant_id/user_gcid flow into the gateway InvokeRequest (RLS + token-usage
// ledger + budget attribution) instead of the process-fixed env tenant.
package modelgatewayclient

import (
	"context"
	"errors"
	"testing"
	"time"

	"google.golang.org/genai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	adkmodel "google.golang.org/adk/model"
)

var errNotFound = errors.New("not found")

// fakeState is a minimal session-state reader for unit-testing the
// BeforeModelCallback glue without standing up a full ADK CallbackContext.
type fakeState struct {
	m map[string]any
}

func (f fakeState) Get(key string) (any, error) {
	v, ok := f.m[key]
	if !ok {
		return nil, errNotFound
	}
	return v, nil
}

// ---------------------------------------------------------------------------
// stampTenantLabels — pure: writes tenant/gcid into req.Config.Labels.
// ---------------------------------------------------------------------------

func TestStampTenantLabels_SetsBothAndInitsConfig(t *testing.T) {
	t.Parallel()
	req := &adkmodel.LLMRequest{} // nil Config
	stampTenantLabels(req, "tenant-B", "gcid-B")
	if req.Config == nil {
		t.Fatal("Config should be initialised")
	}
	if req.Config.Labels[LabelTenantID] != "tenant-B" {
		t.Errorf("tenant label: got %q", req.Config.Labels[LabelTenantID])
	}
	if req.Config.Labels[LabelUserGCID] != "gcid-B" {
		t.Errorf("gcid label: got %q", req.Config.Labels[LabelUserGCID])
	}
}

func TestStampTenantLabels_PreservesExistingLabels(t *testing.T) {
	t.Parallel()
	req := &adkmodel.LLMRequest{
		Config: &genai.GenerateContentConfig{Labels: map[string]string{"keep": "me"}},
	}
	stampTenantLabels(req, "tenant-B", "gcid-B")
	if req.Config.Labels["keep"] != "me" {
		t.Error("existing label should be preserved")
	}
	if req.Config.Labels[LabelTenantID] != "tenant-B" {
		t.Error("tenant label should be added")
	}
}

func TestStampTenantLabels_EmptyIsNoop(t *testing.T) {
	t.Parallel()
	req := &adkmodel.LLMRequest{}
	stampTenantLabels(req, "", "")
	// No tenant/gcid → no labels stamped (client falls back to cfg env).
	if req.Config != nil && (req.Config.Labels[LabelTenantID] != "" || req.Config.Labels[LabelUserGCID] != "") {
		t.Errorf("expected no tenant/gcid labels, got %+v", req.Config.Labels)
	}
}

// ---------------------------------------------------------------------------
// tenantFromLabels — pure: reads tenant/gcid from req.Config.Labels.
// ---------------------------------------------------------------------------

func TestTenantFromLabels_ReadsBoth(t *testing.T) {
	t.Parallel()
	req := &adkmodel.LLMRequest{
		Config: &genai.GenerateContentConfig{Labels: map[string]string{
			LabelTenantID: "tenant-B",
			LabelUserGCID: "gcid-B",
		}},
	}
	tid, gcid := tenantFromLabels(req)
	if tid != "tenant-B" || gcid != "gcid-B" {
		t.Errorf("got tenant=%q gcid=%q", tid, gcid)
	}
}

func TestTenantFromLabels_EmptyWhenAbsent(t *testing.T) {
	t.Parallel()
	if tid, gcid := tenantFromLabels(&adkmodel.LLMRequest{}); tid != "" || gcid != "" {
		t.Errorf("expected empty, got tenant=%q gcid=%q", tid, gcid)
	}
}

// ---------------------------------------------------------------------------
// applyTenantFromState — BeforeModelCallback glue (state → labels).
// ---------------------------------------------------------------------------

func TestApplyTenantFromState_StampsFromState(t *testing.T) {
	t.Parallel()
	req := &adkmodel.LLMRequest{}
	st := fakeState{m: map[string]any{"tenant_id": "tenant-from-state", "user_gcid": "gcid-from-state"}}
	applyTenantFromState(st, req, nil)
	if req.Config.Labels[LabelTenantID] != "tenant-from-state" {
		t.Errorf("tenant: got %q", req.Config.Labels[LabelTenantID])
	}
	if req.Config.Labels[LabelUserGCID] != "gcid-from-state" {
		t.Errorf("gcid: got %q", req.Config.Labels[LabelUserGCID])
	}
}

func TestApplyTenantFromState_MissingStateIsNoop(t *testing.T) {
	t.Parallel()
	req := &adkmodel.LLMRequest{}
	applyTenantFromState(fakeState{m: map[string]any{}}, req, nil)
	if req.Config != nil && req.Config.Labels[LabelTenantID] != "" {
		t.Error("no state → no tenant label")
	}
}

// ---------------------------------------------------------------------------
// ActionCodeResolver — per-request mana action_code stamping (ADR-177 §5).
// ---------------------------------------------------------------------------

func TestApplyTenantFromState_StampsActionCodeFromResolver(t *testing.T) {
	t.Parallel()
	req := &adkmodel.LLMRequest{}
	st := fakeState{m: map[string]any{"tenant_id": "t", "user_gcid": "g", "mana_tier": "premium"}}
	resolver := func(get func(key string) string) string {
		// mirror chora-consumption ChatTurnActionCodeForTier
		return "familiar_chat_turn_" + get("mana_tier")
	}
	applyTenantFromState(st, req, resolver)
	if got := req.Config.Labels[LabelActionCode]; got != "familiar_chat_turn_premium" {
		t.Errorf("action_code label: got %q want familiar_chat_turn_premium", got)
	}
	if got := actionCodeFromLabels(req); got != "familiar_chat_turn_premium" {
		t.Errorf("actionCodeFromLabels: got %q", got)
	}
}

func TestApplyTenantFromState_NilResolverLeavesActionCodeUnset(t *testing.T) {
	t.Parallel()
	req := &adkmodel.LLMRequest{}
	st := fakeState{m: map[string]any{"tenant_id": "t", "user_gcid": "g"}}
	applyTenantFromState(st, req, nil)
	if got := actionCodeFromLabels(req); got != "" {
		t.Errorf("nil resolver → no action_code label; got %q", got)
	}
}

func TestApplyTenantFromState_EmptyResolverResultSkipsLabel(t *testing.T) {
	t.Parallel()
	req := &adkmodel.LLMRequest{}
	st := fakeState{m: map[string]any{"tenant_id": "t", "user_gcid": "g"}}
	applyTenantFromState(st, req, func(func(string) string) string { return "" })
	if got := actionCodeFromLabels(req); got != "" {
		t.Errorf("empty resolver result → no label; got %q", got)
	}
}

func TestNewTenantPropagationPlugin_WithActionCodeResolverBuilds(t *testing.T) {
	t.Parallel()
	p, err := NewTenantPropagationPlugin("familiar", WithActionCodeResolver(func(func(string) string) string { return "familiar_chat_turn_basic" }))
	if err != nil {
		t.Fatalf("NewTenantPropagationPlugin(opts): %v", err)
	}
	if p == nil {
		t.Fatal("nil plugin")
	}
}

// ---------------------------------------------------------------------------
// NewTenantPropagationPlugin — constructs a named plugin with a
// BeforeModelCallback.
// ---------------------------------------------------------------------------

func TestNewTenantPropagationPlugin_Builds(t *testing.T) {
	t.Parallel()
	p, err := NewTenantPropagationPlugin("qgen")
	if err != nil {
		t.Fatalf("NewTenantPropagationPlugin: %v", err)
	}
	if p == nil {
		t.Fatal("nil plugin")
	}
	if p.BeforeModelCallback() == nil {
		t.Error("expected a BeforeModelCallback")
	}
}

// ---------------------------------------------------------------------------
// GenerateContent — per-request tenant override + env fallback (bufconn).
// ---------------------------------------------------------------------------

func TestGenerateContent_PerRequestTenantOverridesEnv(t *testing.T) {
	t.Parallel()
	fake := &fakeServer{}
	llm := mustNewClient(t, fake, nil) // cfg env tenant = 01k...ten

	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "Q?"}}}},
		Config: &genai.GenerateContentConfig{Labels: map[string]string{
			LabelTenantID: "tenant-PER-REQUEST",
			LabelUserGCID: "gcid-PER-REQUEST",
		}},
	}
	for _, err := range llm.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}
	if fake.lastReq.TenantId != "tenant-PER-REQUEST" {
		t.Errorf("tenant_id: got %q, want per-request override", fake.lastReq.TenantId)
	}
	if fake.lastReq.Gcid != "gcid-PER-REQUEST" {
		t.Errorf("gcid: got %q, want per-request override", fake.lastReq.Gcid)
	}
}

func TestGenerateContent_FallsBackToCfgTenantWhenNoLabels(t *testing.T) {
	t.Parallel()
	fake := &fakeServer{}
	llm := mustNewClient(t, fake, nil)

	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "Q?"}}}},
		Config:   &genai.GenerateContentConfig{}, // no tenant labels
	}
	for _, err := range llm.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}
	if fake.lastReq.TenantId != "01k0000000000000000000ten" {
		t.Errorf("tenant_id: got %q, want cfg env fallback", fake.lastReq.TenantId)
	}
	if fake.lastReq.Gcid != "01k0000000000000000000gcid" {
		t.Errorf("gcid: got %q, want cfg env fallback", fake.lastReq.Gcid)
	}
}

// tenant labels MUST NOT leak into the gateway GenerationConfig (they're
// client-side routing only).
func TestGenerateContent_TenantLabelsNotForwardedInGenerationConfig(t *testing.T) {
	t.Parallel()
	fake := &fakeServer{}
	llm := mustNewClient(t, fake, func(c *Config) {
		c.dialOptsExtra = append([]grpc.DialOption{
			grpc.WithTransportCredentials(insecure.NewCredentials()),
		}, c.dialOptsExtra...)
		c.CallTimeout = 5 * time.Second
	})
	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "Q?"}}}},
		Config: &genai.GenerateContentConfig{Labels: map[string]string{
			LabelTenantID: "tenant-X",
		}},
	}
	for _, err := range llm.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}
	if fake.lastReq.GenerationConfig != nil {
		if _, ok := fake.lastReq.GenerationConfig.AsMap()["labels"]; ok {
			t.Error("tenant labels must not be forwarded in GenerationConfig")
		}
	}
}

// ---------------------------------------------------------------------------
// ADR-254 D7 / R22: the dispatch idempotency key rides session state -> label
// -> InvokeRequest so a redelivered dispatch bills once at the gateway.
// ---------------------------------------------------------------------------

func TestApplyTenantFromState_CopiesTheDispatchKeyLabel(t *testing.T) {
	t.Parallel()
	req := &adkmodel.LLMRequest{}
	applyTenantFromState(fakeState{m: map[string]any{
		"tenant_id":                "t1",
		"user_gcid":                "g1",
		"dispatch_idempotency_key": "agent_dispatch.oe_evaluate.sub-1:q-1:1",
	}}, req, nil)
	if got := req.Config.Labels[LabelDispatchIdempotencyKey]; got != "agent_dispatch.oe_evaluate.sub-1:q-1:1" {
		t.Errorf("label %s = %q, want the session-state key", LabelDispatchIdempotencyKey, got)
	}
}

func TestApplyTenantFromState_NoDispatchKeyStampsNothing(t *testing.T) {
	t.Parallel()
	req := &adkmodel.LLMRequest{}
	applyTenantFromState(fakeState{m: map[string]any{"tenant_id": "t1", "user_gcid": "g1"}}, req, nil)
	if _, ok := req.Config.Labels[LabelDispatchIdempotencyKey]; ok {
		t.Errorf("a non-dispatched call must not carry %s", LabelDispatchIdempotencyKey)
	}
}
