// Package labeledgemini implements ChoraLabeledGemini — the thin language-
// side wrapper around an LLM that stamps cost-attribution labels per request,
// routes per-tenant LoRA adapters, emits cost-tracking events, and wraps
// every call in an OTel span with OpenInference semantic conventions.
//
// Per ADR-146 (Model Broker full retirement): this is the canonical Chora
// "ChoraLabeledGemini" extension referenced in §5 of ADR-146. The Model
// Broker 3 services are RETIRED — this package + manaplugin + the
// cloud-neutral composable primitives are the replacement.
//
// Hexagonal boundary:
//
//   - LabeledLLM is the inbound port the ADK Go agent code depends on.
//   - FakeLabeledLLM is the test adapter.
//   - LoRARouter is the outbound port for per-tenant adapter selection.
//   - EventSink is the outbound port for cost-tracking event emission.
//
// Per feedback_no_inline_config: NO endpoint URLs, project IDs, region
// strings, or label defaults are hard-coded in this package. Configuration
// flows in via Config or environment-driven helpers in sibling packages.
package labeledgemini

import (
	"context"
	"errors"
)

// LoRARouter resolves a tenant_id to an optional per-tenant Gemma LoRA
// adapter resource path. When the returned adapter is non-empty, the
// LabeledLLM SHOULD route the request to the LoRA-adapted Gemma endpoint
// per ADR-146 §6 (multi-LoRA adapter endpoint pattern).
//
// Implementations typically wrap a domain registry (chora_tenancy adapter
// inventory). Returning an empty AdapterPath means "no per-tenant adapter;
// use the base model".
//
// The error channel is reserved for transient lookup failures (e.g.,
// outbox/database hiccup). Callers MUST decide whether to fail-open or
// fail-closed per ADR-146 §Resilience.
type LoRARouter interface {
	ResolveAdapter(ctx context.Context, tenantID string) (AdapterRef, error)
}

// AdapterRef carries a per-tenant LoRA adapter resource path plus version
// metadata. AdapterPath empty => no per-tenant adapter (use base model).
type AdapterRef struct {
	// AdapterPath is the per-tenant adapter identifier
	// (e.g., "tenant-acme-v3"). Empty for the base model.
	AdapterPath string

	// Version is the adapter version number — surfaced in trace
	// attributes for IMDA D2 transparency / provenance ledger.
	Version int
}

// InMemoryLoRARouter is a deterministic test adapter for LoRARouter.
// Construct via NewInMemoryLoRARouter; mutate via Set / Delete.
type InMemoryLoRARouter struct {
	adapters map[string]AdapterRef
	err      error
}

// NewInMemoryLoRARouter constructs an empty in-memory router.
func NewInMemoryLoRARouter() *InMemoryLoRARouter {
	return &InMemoryLoRARouter{adapters: map[string]AdapterRef{}}
}

// Set registers an adapter for tenantID. Empty AdapterPath is allowed
// and means "explicitly resolved to base model" (distinct from "tenant
// not registered" which returns the zero AdapterRef).
func (r *InMemoryLoRARouter) Set(tenantID string, ref AdapterRef) {
	r.adapters[tenantID] = ref
}

// SetError makes every ResolveAdapter call return err. Used by tests to
// simulate lookup failures.
func (r *InMemoryLoRARouter) SetError(err error) { r.err = err }

// ResolveAdapter implements LoRARouter.
func (r *InMemoryLoRARouter) ResolveAdapter(_ context.Context, tenantID string) (AdapterRef, error) {
	if r.err != nil {
		return AdapterRef{}, r.err
	}
	if tenantID == "" {
		return AdapterRef{}, errors.New("labeledgemini: tenantID required")
	}
	if ref, ok := r.adapters[tenantID]; ok {
		return ref, nil
	}
	// Unknown tenant => zero AdapterRef (base model). NOT an error —
	// the bulk of tenants use the base model.
	return AdapterRef{}, nil
}

// NoopLoRARouter always resolves to the base model. Construct via
// NewNoopLoRARouter; useful as a default when LoRA is not yet wired.
type NoopLoRARouter struct{}

// NewNoopLoRARouter constructs the no-op router.
func NewNoopLoRARouter() NoopLoRARouter { return NoopLoRARouter{} }

// ResolveAdapter always returns the zero AdapterRef.
func (NoopLoRARouter) ResolveAdapter(_ context.Context, _ string) (AdapterRef, error) {
	return AdapterRef{}, nil
}
