// Package instancedispatch implements the hosted-runtime variant of the
// session-time per-instance configuration pattern (ADR-147 §7).
//
// Per discovery #4 from POC week-1: the hosted ADK runtime bypasses
// `agent.Loader.LoadAgent(appName)` (the stream_query controller calls only
// `RootAgent()` and hardcodes AppName to the engine ID). So multi-instance
// dispatch on a hosted runtime MUST happen via per-session `session.State()`
// — NOT via `session.AppName()`.
//
// This package is generic over the instance discriminator. Three crews use it
// today (per ADR-147 §7):
//
//	Familiar Companion       — StateKey="familiar_id"
//	Course Planner           — StateKey="tenant_id"
//	A2A External Mediator    — StateKey="familiar_exposure_grant_id"
//
// Each crew provides a Resolver closure that maps the discriminator value to
// an InstanceConfig (composed Instruction string + AllowedTools allowlist).
// The plugin handles:
//
//   - BeforeRunCallback: refuses the turn if the state key is absent or the
//     resolver fails (IMDA D3 discipline mirroring manaplugin's
//     tenant_id/user_gcid refusal).
//   - BeforeModelCallback: filters `req.Tools` + `req.Config.Tools[].FunctionDeclarations`
//     down to the resolver's AllowedTools per turn.
//
// Pair with NewInstructionProvider — wired into `llmagent.Config.InstructionProvider`
// — to produce the per-instance system prompt per turn. The plugin and the
// instruction provider share the same Resolver so both stay coherent.
//
// A per-session loader (the agent service's own `internal/agent/loader.go`)
// remains the canonical pattern for direct REST (`adkrest`) deploy paths,
// where the controller honours `req.AppName`. The two variants compose:
// same Resolver shape, different dispatch surface.
package instancedispatch

import (
	"context"
	"errors"
	"fmt"

	"google.golang.org/adk/model"
)

// InstanceConfig is the per-turn payload the plugin needs to compose the
// per-instance system prompt and to filter the tool surface.
//
// Crew-side Resolvers map their domain config (e.g., FamiliarConfig,
// TenantCoursePlannerRules, FamiliarExposureGrant) onto this shape.
type InstanceConfig struct {
	// Instruction is the fully-composed system prompt for this turn. The
	// llmagent's InstructionProvider returns this string directly.
	Instruction string

	// AllowedTools is the ordered list of tool names the LLM is allowed to
	// invoke for this turn. Anything not in the list is stripped from the
	// LLMRequest by BeforeModelCallback.
	AllowedTools []string
}

// Resolver maps an instance discriminator (e.g., familiar_id) to the per-turn
// InstanceConfig.
//
// Implementations typically wrap a domain registry (skill registry, tenant
// rules table, exposure grant lookup). Errors are propagated by the plugin to
// refuse the turn — the agent never runs against a stale or missing config.
type Resolver func(ctx context.Context, instanceID string) (InstanceConfig, error)

// Config wires the plugin + InstructionProvider for a specific crew.
type Config struct {
	// Resolver is the lookup function. Required.
	Resolver Resolver

	// StateKey is the session.State() key the caller writes the instance
	// discriminator into at session creation. Examples: "familiar_id",
	// "tenant_id", "familiar_exposure_grant_id". Required.
	StateKey string
}

// validate ensures Config has both required fields. Called by New and
// NewInstructionProvider before returning.
func (c Config) validate() error {
	if c.Resolver == nil {
		return errors.New("instancedispatch: Resolver is required")
	}
	if c.StateKey == "" {
		return errors.New("instancedispatch: StateKey is required (e.g., \"familiar_id\")")
	}
	return nil
}

// readonlyStateGetter is the minimum the package needs from
// session.ReadonlyState. Lets the pure functions stay testable without
// implementing the full state interface.
type readonlyStateGetter interface {
	Get(string) (any, error)
}

// ResolveInstanceFromState reads the discriminator from state[stateKey] and
// calls resolver. Pure function — no plugin context needed.
//
// Returns an error if:
//   - stateKey or resolver are zero (programmer error)
//   - state doesn't contain stateKey
//   - state[stateKey] is not a non-empty string
//   - resolver returns an error
func ResolveInstanceFromState(
	ctx context.Context,
	state readonlyStateGetter,
	stateKey string,
	resolver Resolver,
) (InstanceConfig, error) {
	if stateKey == "" {
		return InstanceConfig{}, errors.New("instancedispatch: stateKey must not be empty")
	}
	if resolver == nil {
		return InstanceConfig{}, errors.New("instancedispatch: resolver must not be nil")
	}
	raw, err := state.Get(stateKey)
	if err != nil {
		return InstanceConfig{}, fmt.Errorf(
			"session state missing %q (required for per-instance dispatch; "+
				"caller must pass it at async_create_session): %w", stateKey, err)
	}
	id, ok := raw.(string)
	if !ok {
		return InstanceConfig{}, fmt.Errorf(
			"session state %q must be a string; got %T", stateKey, raw)
	}
	if id == "" {
		return InstanceConfig{}, fmt.Errorf(
			"session state %q is empty; refusing call", stateKey)
	}
	ctx = WithInstanceID(ctx, id)
	// Stamp the caller's identity (tenant_id + user_gcid) from session state
	// onto ctx so resolvers that need it — e.g. the Familiar gRPC registry
	// calling chora-consumption ResolveFamiliarConfig — can recover it without
	// widening the fixed Resolver(ctx, instanceID) signature. Additive +
	// best-effort: absent/non-string keys are skipped; crews that don't read
	// the values are unaffected.
	ctx = stampIdentityFromState(ctx, state)
	out, err := resolver(ctx, id)
	if err != nil {
		return InstanceConfig{}, fmt.Errorf("resolve %s=%s: %w", stateKey, safePrefix(id, 8), err)
	}
	return out, nil
}

// -----------------------------------------------------------------------------
// Request-identity ctx propagation (well-known session-state keys)
//
// The Resolver signature is fixed at (ctx, instanceID), but some resolvers need
// the caller's tenant_id + user_gcid (written into session.State() at
// async_create_session — the same keys manaplugin reads). ResolveInstanceFromState
// stamps them onto the resolver's ctx via the accessors below.
// -----------------------------------------------------------------------------

type identityCtxKey int

const (
	ctxKeyTenantID identityCtxKey = iota
	ctxKeyUserGCID
	ctxKeyFamiliarConfig
	ctxKeyInstanceID
	ctxKeyPromptOverrides
)

// Canonical session-state keys (caller identity + optional per-instance config).
const (
	stateKeyTenantID       = "tenant_id"
	stateKeyUserGCID       = "user_gcid"
	stateKeyFamiliarConfig = "familiar_config"
	// ADR-197 P3 (CHO-2368): the registry-resolved prompt-override map
	// (JSON string, segment_id -> body) the session creator injected at
	// async_create_session - same delivery contract the orchestrator's
	// _build_session_state uses for qgen/OE.
	stateKeyPromptOverrides = "prompt_overrides_json"
)

// WithRequestIdentity stamps tenant_id + user_gcid onto ctx. Exported so tests
// and non-dispatch callers can seed the same ctx a resolver sees. Empty values
// are not stamped.
func WithRequestIdentity(ctx context.Context, tenantID, userGCID string) context.Context {
	if tenantID != "" {
		ctx = context.WithValue(ctx, ctxKeyTenantID, tenantID)
	}
	if userGCID != "" {
		ctx = context.WithValue(ctx, ctxKeyUserGCID, userGCID)
	}
	return ctx
}

// WithInstanceID stamps the per-instance discriminator (e.g. familiar_id)
// onto ctx so tool handlers can recover WHICH instance the turn belongs to
// without trusting an LLM-supplied parameter (CHO-2013 P1.B).
func WithInstanceID(ctx context.Context, instanceID string) context.Context {
	if instanceID != "" {
		ctx = context.WithValue(ctx, ctxKeyInstanceID, instanceID)
	}
	return ctx
}

// InstanceIDFromContext returns the discriminator stamped by WithInstanceID
// (via ResolveInstanceFromState), or "".
func InstanceIDFromContext(ctx context.Context) string {
	v, _ := ctx.Value(ctxKeyInstanceID).(string)
	return v
}

// TenantIDFromContext returns the tenant_id stamped by WithRequestIdentity, or "".
func TenantIDFromContext(ctx context.Context) string {
	v, _ := ctx.Value(ctxKeyTenantID).(string)
	return v
}

// UserGCIDFromContext returns the user_gcid stamped by WithRequestIdentity, or "".
func UserGCIDFromContext(ctx context.Context) string {
	v, _ := ctx.Value(ctxKeyUserGCID).(string)
	return v
}

// WithFamiliarConfigJSON stamps an opaque per-instance config payload (e.g. the
// protojson FamiliarInstanceConfig the caller wrote into session state at
// async_create_session) onto ctx, so a resolver can read its full config from
// state instead of calling back to the owning service. Empty values are not
// stamped. Exported for tests + non-dispatch seeders.
func WithFamiliarConfigJSON(ctx context.Context, cfgJSON string) context.Context {
	if cfgJSON != "" {
		ctx = context.WithValue(ctx, ctxKeyFamiliarConfig, cfgJSON)
	}
	return ctx
}

// FamiliarConfigJSONFromContext returns the config payload stamped by
// WithFamiliarConfigJSON / stampIdentityFromState, or "".
func FamiliarConfigJSONFromContext(ctx context.Context) string {
	v, _ := ctx.Value(ctxKeyFamiliarConfig).(string)
	return v
}

// WithPromptOverridesJSON stamps the ADR-197 prompt-override map (JSON string,
// segment_id -> body) onto ctx so an instance resolver can thread it into its
// prompt composer. Empty values are not stamped. Exported for tests +
// non-dispatch seeders.
func WithPromptOverridesJSON(ctx context.Context, overridesJSON string) context.Context {
	if overridesJSON != "" {
		ctx = context.WithValue(ctx, ctxKeyPromptOverrides, overridesJSON)
	}
	return ctx
}

// PromptOverridesJSONFromContext returns the override payload stamped by
// WithPromptOverridesJSON / stampIdentityFromState, or "".
func PromptOverridesJSONFromContext(ctx context.Context) string {
	v, _ := ctx.Value(ctxKeyPromptOverrides).(string)
	return v
}

// stampIdentityFromState reads the canonical identity + per-instance config
// keys from state (best-effort) and returns ctx enriched with them.
func stampIdentityFromState(ctx context.Context, state readonlyStateGetter) context.Context {
	tenantID, _ := stringFromState(state, stateKeyTenantID)
	userGCID, _ := stringFromState(state, stateKeyUserGCID)
	ctx = WithRequestIdentity(ctx, tenantID, userGCID)
	if cfgJSON, ok := stringFromState(state, stateKeyFamiliarConfig); ok {
		ctx = WithFamiliarConfigJSON(ctx, cfgJSON)
	}
	if ovrJSON, ok := stringFromState(state, stateKeyPromptOverrides); ok {
		ctx = WithPromptOverridesJSON(ctx, ovrJSON)
	}
	return ctx
}

func stringFromState(state readonlyStateGetter, key string) (string, bool) {
	raw, err := state.Get(key)
	if err != nil {
		return "", false
	}
	s, ok := raw.(string)
	return s, ok
}

// ApplyToolFilter mutates the LLMRequest in-place to retain only tools whose
// declared name appears in allowed.
//
// Both surfaces are filtered:
//   - req.Config.Tools[*].FunctionDeclarations — what the LLM sees as callable
//   - req.Tools — the Go-side implementation map keyed by tool name
//
// Safe with nil req, nil req.Config, empty allowed (drops everything).
func ApplyToolFilter(req *model.LLMRequest, allowed []string) {
	if req == nil {
		return
	}
	allowSet := make(map[string]struct{}, len(allowed))
	for _, name := range allowed {
		allowSet[name] = struct{}{}
	}

	// Filter the Go-side impl map.
	for name := range req.Tools {
		if _, ok := allowSet[name]; !ok {
			delete(req.Tools, name)
		}
	}

	// Filter the LLM-visible function declarations across all tool groups, then
	// drop any function Tool whose declarations are now empty. Gemini rejects an
	// empty Tool ("tools[0].tool_type: required one_of 'tool_type' must have one
	// initialized field", INVALID_ARGUMENT) — so an instance with an empty
	// allow-list must send NO tools, not an empty tool group.
	if req.Config == nil {
		return
	}
	kept := req.Config.Tools[:0]
	for _, tl := range req.Config.Tools {
		if tl == nil {
			continue
		}
		if tl.FunctionDeclarations == nil {
			// Non-function tool (e.g. retrieval / search) — no declared name to
			// filter on; preserve it untouched.
			kept = append(kept, tl)
			continue
		}
		retained := tl.FunctionDeclarations[:0]
		for _, decl := range tl.FunctionDeclarations {
			if _, ok := allowSet[decl.Name]; ok {
				retained = append(retained, decl)
			}
		}
		tl.FunctionDeclarations = retained
		if len(retained) > 0 {
			kept = append(kept, tl)
		}
	}
	req.Config.Tools = kept
}

// safePrefix returns the first n runes of s with an ellipsis when truncated.
// Mirrors manaplugin.safePrefix — used for log/error formatting to avoid
// leaking full instance UUIDs into error messages.
func safePrefix(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
