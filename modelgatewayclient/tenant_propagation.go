// tenant_propagation.go — ADR-169 per-request tenant attribution.
//
// Before ADR-169 the gateway client pinned InvokeRequest.TenantId / Gcid to
// the process-wide CHORA_GATEWAY_TENANT_ID / CHORA_GATEWAY_GCID env vars (set
// once at construction). That is correct for a single-tenant verify pod but
// MIS-ATTRIBUTES every call to one tenant on a permanent multi-tenant
// deployment — the gateway uses TenantId for RLS, the token-usage ledger, and
// per-tenant budget enforcement, so all tenants would bill (and be RLS-scoped)
// as the env tenant.
//
// The orchestrator already threads the REQUESTING tenant into ADK session
// state (reasoning_engine_executor._build_session_state -> state["tenant_id"]
// + state["user_gcid"]), and manaplugin REFUSES any run whose state lacks
// them. We bridge state -> the per-call gateway request via a BeforeModelCallback
// that stamps tenant_id / user_gcid into req.Config.Labels (genai's
// "user-defined metadata to break down billed charges" — semantically exactly
// this); GenerateContent reads those labels and overrides the env defaults,
// falling back to the env values only when a label is absent (e.g. a direct
// non-orchestrator caller). The labels are consumed client-side only — they are
// NOT forwarded to the gateway in GenerationConfig.
package modelgatewayclient

import (
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	adkmodel "google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
)

// Session-state keys the orchestrator threads + manaplugin guarantees present,
// plus the dispatch idempotency key agentdispatch.Request.SessionState sets on
// every event-dispatched run (ADR-254 D7 / R22; absent on other callers).
const (
	stateKeyTenantID               = "tenant_id"
	stateKeyUserGCID               = "user_gcid"
	stateKeyDispatchIdempotencyKey = "dispatch_idempotency_key"
)

// genai.Config.Labels keys carrying the per-request tenant scope + per-turn
// mana action_code from session state to GenerateContent. Underscored
// (label-safe) even though they stay client-side. Read by tenantFromLabels
// / actionCodeFromLabels; written by stampTenantLabels / stampActionCodeLabel.
const (
	LabelTenantID   = "chora_tenant_id"
	LabelUserGCID   = "chora_user_gcid"
	LabelActionCode = "chora_action_code"
	// LabelDispatchIdempotencyKey carries the event-dispatch request's
	// idempotency_key from session state to InvokeRequest.dispatch_idempotency_key
	// (ADR-254 D7 / R22). Absent on a non-dispatched call, so nothing is stamped.
	LabelDispatchIdempotencyKey = "chora_dispatch_idempotency_key"
)

// stateReader is the minimal session-state read surface (matches
// session.ReadonlyState / session.State Get). Mirrors manaplugin.getStateString.
type stateReader interface {
	Get(key string) (any, error)
}

// stampTenantLabels writes tenant/gcid into req.Config.Labels (initialising
// Config + Labels as needed). Empty values are skipped so the client falls
// back to its env-configured default for that field.
func stampTenantLabels(req *adkmodel.LLMRequest, tenantID, gcid string) {
	if req == nil || (tenantID == "" && gcid == "") {
		return
	}
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	if req.Config.Labels == nil {
		req.Config.Labels = map[string]string{}
	}
	if tenantID != "" {
		req.Config.Labels[LabelTenantID] = tenantID
	}
	if gcid != "" {
		req.Config.Labels[LabelUserGCID] = gcid
	}
}

// tenantFromLabels reads the per-request tenant/gcid stamped by
// stampTenantLabels. Returns empty strings when absent (caller falls back to
// the env-configured cfg values).
func tenantFromLabels(req *adkmodel.LLMRequest) (tenantID, gcid string) {
	if req == nil || req.Config == nil || req.Config.Labels == nil {
		return "", ""
	}
	return req.Config.Labels[LabelTenantID], req.Config.Labels[LabelUserGCID]
}

// stampActionCodeLabel writes a per-request action_code override into
// req.Config.Labels (ADR-177 §5). Empty is skipped so GenerateContent falls
// back to the client's static cfg.ActionCode.
func stampActionCodeLabel(req *adkmodel.LLMRequest, actionCode string) {
	if req == nil || actionCode == "" {
		return
	}
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	if req.Config.Labels == nil {
		req.Config.Labels = map[string]string{}
	}
	req.Config.Labels[LabelActionCode] = actionCode
}

// stampDispatchKeyLabel writes the dispatch idempotency key into
// req.Config.Labels. Empty is skipped: a non-dispatched call stamps nothing and
// the gateway takes no keyed claim for it.
func stampDispatchKeyLabel(req *adkmodel.LLMRequest, key string) {
	if req == nil || key == "" {
		return
	}
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	if req.Config.Labels == nil {
		req.Config.Labels = map[string]string{}
	}
	req.Config.Labels[LabelDispatchIdempotencyKey] = key
}

// dispatchKeyFromLabels reads the dispatch idempotency key stamped by
// stampDispatchKeyLabel. Empty when the call was not a dispatch.
func dispatchKeyFromLabels(req *adkmodel.LLMRequest) string {
	if req == nil || req.Config == nil || req.Config.Labels == nil {
		return ""
	}
	return req.Config.Labels[LabelDispatchIdempotencyKey]
}

// actionCodeFromLabels reads the per-request action_code override stamped by
// stampActionCodeLabel. Empty ⇒ GenerateContent uses the static cfg.ActionCode.
func actionCodeFromLabels(req *adkmodel.LLMRequest) string {
	if req == nil || req.Config == nil || req.Config.Labels == nil {
		return ""
	}
	return req.Config.Labels[LabelActionCode]
}

// ActionCodeResolver computes a per-request mana action_code from session state
// (e.g. familiar's per-tier `familiar_chat_turn_{tier}` read off `mana_tier`).
// It receives a string accessor over the session state (empty string on miss)
// so callers need not depend on the internal state-reader type. Return "" to
// leave the client on its static cfg.ActionCode.
type ActionCodeResolver func(getState func(key string) string) string

// PropagationOption configures NewTenantPropagationPlugin.
type PropagationOption func(*propagationConfig)

type propagationConfig struct {
	actionCodeResolver ActionCodeResolver
}

// WithActionCodeResolver installs a per-request action_code resolver — used by
// crews whose action_code is dynamic per session (familiar's mana tier). Crews
// with a single static action_code set Config.ActionCode on the client instead
// and need no resolver.
func WithActionCodeResolver(r ActionCodeResolver) PropagationOption {
	return func(c *propagationConfig) { c.actionCodeResolver = r }
}

// applyTenantFromState copies tenant_id/user_gcid (and, when a resolver is set,
// the per-request action_code) from session state into req.Config.Labels. The
// BeforeModelCallback glue; factored out for unit testing without a full ADK
// CallbackContext.
func applyTenantFromState(state stateReader, req *adkmodel.LLMRequest, resolver ActionCodeResolver) {
	stampTenantLabels(req, readState(state, stateKeyTenantID), readState(state, stateKeyUserGCID))
	stampDispatchKeyLabel(req, readState(state, stateKeyDispatchIdempotencyKey))
	if resolver != nil {
		stampActionCodeLabel(req, resolver(func(key string) string { return readState(state, key) }))
	}
}

func readState(state stateReader, key string) string {
	v, err := state.Get(key)
	if err != nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

// NewTenantPropagationPlugin returns an ADK plugin whose BeforeModelCallback
// stamps the per-request tenant_id/user_gcid (from session state) into each
// LLMRequest so the gateway client attributes RLS + ledger + budget to the
// REQUESTING tenant. With WithActionCodeResolver it ALSO stamps a per-request
// mana action_code (ADR-177 §5) — used by crews whose code is dynamic per
// session (familiar's tier). Wire alongside manaplugin in the launcher
// PluginConfig.
//
// Register this on EVERY crew that issues LLM calls through the gateway client
// on a permanent (multi-tenant) deployment (ADR-169).
func NewTenantPropagationPlugin(crewKind string, opts ...PropagationOption) (*plugin.Plugin, error) {
	var cfg propagationConfig
	for _, opt := range opts {
		opt(&cfg)
	}
	return plugin.New(plugin.Config{
		Name: "chora_gateway_tenant_propagation_" + crewKind,
		BeforeModelCallback: func(ctx agent.CallbackContext, req *adkmodel.LLMRequest) (*adkmodel.LLMResponse, error) {
			applyTenantFromState(ctx.ReadonlyState(), req, cfg.actionCodeResolver)
			return nil, nil
		},
	})
}
