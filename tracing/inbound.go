// inbound.go — continue a distributed trace from an inbound W3C traceparent.
//
// Why this exists (CR qgen Phase B2, 2026-06-01 — per-workload single trace):
//
// The Python LangGraph orchestrator (qgen_crew.py) drives the qgen agents via
// the reasoning-engine executor. Today the orchestrator→agent hop carries NO
// trace context, so each ADK agent starts a FRESH root trace tree — a single
// authoring workflow shatters into 3+ disconnected trace backend traces
// (orchestrator / qgen_question / qgen_critic / gateway). The agent→gateway
// hop already links (modelgatewayclient injects the active span via otelgrpc);
// the missing link is orchestrator→agent.
//
// The orchestrator now stamps a W3C `traceparent` (and optional `tracestate`)
// into the ADK session state the executor builds (key "traceparent" — see
// reasoning_engine_executor._build_session_state). This file parses that
// inbound string into an OTel remote SpanContext so the agent can CONTINUE the
// orchestrator's trace.
//
// LIMITATION (documented, load-bearing): the ADK runtime creates the agent's
// ROOT `invoke_agent` span from the inbound HTTP request context BEFORE any
// plugin / instruction-provider callback fires, and the ADK agentengine REST
// server does NOT extract a propagator carrier from the request (verified
// against google.golang.org/adk v1.2.1 server/agentengine/controllers/
// agent_engine.go — req.Context() is passed straight through). The earliest
// hook with BOTH the active local span AND access to session.State() is the
// BeforeAgentCallback (runs after agent.go:StartInvokeAgentSpan). Plugin
// callbacks cannot mutate the context the runner passes to agent.Run, so the
// ADK root span is NOT re-parentable from agent code.
//
// Therefore full parent→child re-parenting of the ADK root span across the
// orchestrator→agent boundary is NOT achievable with the current ADK. The best
// available linkage is a span LINK: the agent emits a child span seeded from
// the inbound remote SpanContext and links it to the local ADK trace (and
// vice-versa), so trace backend's UI connects the two trace trees bidirectionally
// for a given workflow. AddInboundLink + StartLinkedInboundSpan below implement
// that. When the ADK server gains request-context propagation (or exposes a
// pre-root hook), ContextFromTraceparent can directly seed the root via
// ContextWithRemoteSpanContext for true single-tree parenting — the parse logic
// is already correct for that path.
package tracing

import (
	"context"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
)

// inboundCarrierKey is the W3C traceparent header name, reused as the session-
// state key the Python executor writes (reasoning_engine_executor.py
// _build_session_state -> state["traceparent"]). Kept identical to the HTTP
// header name so the standard propagation.TraceContext propagator can parse it
// from a synthetic single-entry carrier.
const inboundCarrierKey = "traceparent"

// inboundTracestateKey mirrors the W3C tracestate header name. Optional.
const inboundTracestateKey = "tracestate"

// SpanContextFromTraceparent parses a W3C `traceparent` (and optional
// `tracestate`) string into an OTel remote SpanContext.
//
// It delegates to the globally-registered propagation.TraceContext propagator
// over a synthetic single-entry carrier, so the parse rules (version, hex
// lengths, flags, all-zero rejection) exactly match the wire spec and the
// project's inbound HTTP/gRPC path. tracing.Init installs that propagator
// globally; we also fall back to a local propagator instance so this helper is
// usable before Init (and in unit tests).
//
// Returns the parsed SpanContext and ok=true when the traceparent is well-
// formed and valid; returns the zero SpanContext and ok=false otherwise (a
// safe no-op — the caller keeps its existing context unchanged).
func SpanContextFromTraceparent(traceparent, tracestate string) (trace.SpanContext, bool) {
	if traceparent == "" {
		return trace.SpanContext{}, false
	}
	carrier := propagation.MapCarrier{inboundCarrierKey: traceparent}
	if tracestate != "" {
		carrier[inboundTracestateKey] = tracestate
	}
	prop := otel.GetTextMapPropagator()
	// otel.GetTextMapPropagator() defaults to a no-op when Init hasn't run
	// (e.g. unit tests that don't call tracing.Init). Use a concrete
	// TraceContext propagator in that case so parsing still works.
	if prop == nil {
		prop = propagation.TraceContext{}
	}
	ctx := prop.Extract(context.Background(), carrier)
	sc := trace.SpanContextFromContext(ctx)
	if !sc.IsValid() {
		// The no-op default propagator extracts nothing; retry explicitly
		// with the W3C TraceContext propagator before giving up.
		ctx = (propagation.TraceContext{}).Extract(context.Background(), carrier)
		sc = trace.SpanContextFromContext(ctx)
	}
	if !sc.IsValid() {
		return trace.SpanContext{}, false
	}
	// Mark the SpanContext as remote — it originated in another process
	// (the Python orchestrator), so child spans + links treat it correctly.
	return sc.WithRemote(true), true
}

// ContextFromTraceparent returns a child context that carries the inbound
// orchestrator SpanContext as the active span context. A span subsequently
// started from the returned context becomes a CHILD of the orchestrator span,
// continuing the same distributed trace (single trace tree).
//
// On an empty or malformed traceparent it returns the input ctx UNCHANGED
// (safe no-op) — the agent then runs on its own local trace tree rather than
// crashing or silently corrupting context.
//
// NOTE: with the current ADK runtime this is only effective at hooks that run
// BEFORE the ADK root span is created (none are exposed to agent code today —
// see the package doc LIMITATION). For the in-agent linkage that IS achievable,
// use StartLinkedInboundSpan / AddInboundLink.
func ContextFromTraceparent(ctx context.Context, traceparent, tracestate string) context.Context {
	sc, ok := SpanContextFromTraceparent(traceparent, tracestate)
	if !ok {
		return ctx
	}
	return trace.ContextWithRemoteSpanContext(ctx, sc)
}

// StartLinkedInboundSpan starts a child span named spanName whose parent is the
// active span in ctx (the live ADK invoke_agent root, when called from a
// BeforeAgentCallback) AND which carries a Link to the inbound orchestrator
// SpanContext parsed from traceparent. This is the best-available cross-process
// linkage given that the ADK root span cannot be re-parented from agent code
// (see the package doc LIMITATION).
//
// The returned context carries the new span; the returned func ends it (call
// it when the linked work completes, or immediately if you only need the link
// edge to exist). When traceparent is empty/malformed, a normal child span is
// started with no link (still safe + useful as an explicit per-turn span).
//
// tracerName scopes the OTel tracer (pass the agent's service.name, e.g.
// "qgen_question").
func StartLinkedInboundSpan(
	ctx context.Context,
	tracerName, spanName, traceparent, tracestate string,
) (context.Context, func()) {
	tr := otel.Tracer(tracerName)
	opts := []trace.SpanStartOption{}
	if sc, ok := SpanContextFromTraceparent(traceparent, tracestate); ok {
		opts = append(opts, trace.WithLinks(trace.Link{
			SpanContext: sc,
			Attributes: []attribute.KeyValue{
				attribute.String("chora.trace.link_kind", "orchestrator_inbound"),
			},
		}))
	}
	newCtx, span := tr.Start(ctx, spanName, opts...)
	return newCtx, func() { span.End() }
}

// AddInboundLink attaches a Link to the inbound orchestrator SpanContext onto
// the ALREADY-ACTIVE span in ctx (e.g. the ADK invoke_agent root span, when
// called from a BeforeAgentCallback). Unlike StartLinkedInboundSpan it creates
// no new span — it annotates the existing one, so the agent's root span itself
// references the orchestrator trace.
//
// Returns true when a link was added (valid inbound traceparent + recording
// span), false otherwise (safe no-op).
func AddInboundLink(ctx context.Context, traceparent, tracestate string) bool {
	sc, ok := SpanContextFromTraceparent(traceparent, tracestate)
	if !ok {
		return false
	}
	span := trace.SpanFromContext(ctx)
	if span == nil || !span.IsRecording() {
		return false
	}
	span.AddLink(trace.Link{
		SpanContext: sc,
		Attributes: []attribute.KeyValue{
			attribute.String("chora.trace.link_kind", "orchestrator_inbound"),
		},
	})
	return true
}
