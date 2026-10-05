package tracing

import (
	"context"
	"testing"

	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// withGlobalTracerProvider temporarily installs tp as the global OTel
// TracerProvider (which otel.Tracer(...) reads), runs fn, then restores the
// previous provider. Lets the StartLinkedInboundSpan tests capture spans via a
// recorder-backed provider.
func withGlobalTracerProvider(t *testing.T, tp trace.TracerProvider, fn func()) {
	t.Helper()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	defer otel.SetTracerProvider(prev)
	fn()
}

// A canonical valid W3C traceparent (from the spec examples). trace_id +
// span_id are the load-bearing fields the agent must inherit.
const (
	validTraceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
	validTraceID     = "0af7651916cd43dd8448eb211c80319c"
	validSpanID      = "b7ad6b7169203331"
)

// TestSpanContextFromTraceparent_Valid proves a well-formed traceparent parses
// into a remote SpanContext whose TraceID + SpanID match the wire string.
func TestSpanContextFromTraceparent_Valid(t *testing.T) {
	sc, ok := SpanContextFromTraceparent(validTraceparent, "")
	if !ok {
		t.Fatalf("SpanContextFromTraceparent(%q) ok=false, want true", validTraceparent)
	}
	if got := sc.TraceID().String(); got != validTraceID {
		t.Errorf("TraceID=%q want %q", got, validTraceID)
	}
	if got := sc.SpanID().String(); got != validSpanID {
		t.Errorf("SpanID=%q want %q", got, validSpanID)
	}
	if !sc.IsRemote() {
		t.Errorf("expected IsRemote()=true for an inbound (cross-process) traceparent")
	}
	if !sc.IsSampled() {
		t.Errorf("expected IsSampled()=true (flags=01 in the traceparent)")
	}
}

// TestSpanContextFromTraceparent_Malformed proves bad inputs are a SAFE no-op
// (ok=false, zero SpanContext) rather than a panic or a bogus context.
func TestSpanContextFromTraceparent_Malformed(t *testing.T) {
	cases := []struct {
		name string
		tp   string
	}{
		{"empty", ""},
		{"garbage", "not-a-traceparent"},
		{"truncated", "00-0af7651916cd43dd8448eb211c80319c"},
		{"all-zero-trace-id", "00-00000000000000000000000000000000-b7ad6b7169203331-01"},
		{"all-zero-span-id", "00-0af7651916cd43dd8448eb211c80319c-0000000000000000-01"},
		{"bad-hex", "00-zzf7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			sc, ok := SpanContextFromTraceparent(c.tp, "")
			if ok {
				t.Errorf("SpanContextFromTraceparent(%q) ok=true, want false", c.tp)
			}
			if sc.IsValid() {
				t.Errorf("SpanContextFromTraceparent(%q) returned a valid SpanContext, want zero", c.tp)
			}
		})
	}
}

// TestContextFromTraceparent_Valid proves a span started from the returned
// context is a CHILD of the inbound orchestrator span (same TraceID,
// remote-parent SpanID), i.e. the agent continues the orchestrator's trace.
func TestContextFromTraceparent_Valid(t *testing.T) {
	ctx := ContextFromTraceparent(context.Background(), validTraceparent, "")

	// The remote span context must now be the active span context.
	sc := trace.SpanContextFromContext(ctx)
	if got := sc.TraceID().String(); got != validTraceID {
		t.Fatalf("active TraceID=%q want %q (context did not inherit orchestrator trace)", got, validTraceID)
	}

	// A span started from this context inherits the orchestrator TraceID +
	// is parented to the orchestrator span.
	var out []sdktrace.ReadOnlySpan
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(spanRecorder{out: &out}))
	_, span := tp.Tracer("test").Start(ctx, "agent_root")
	span.End()
	if len(out) != 1 {
		t.Fatalf("expected 1 recorded span, got %d", len(out))
	}
	child := out[0]
	if got := child.SpanContext().TraceID().String(); got != validTraceID {
		t.Errorf("child TraceID=%q want %q", got, validTraceID)
	}
	if got := child.Parent().SpanID().String(); got != validSpanID {
		t.Errorf("child parent SpanID=%q want %q (not parented to orchestrator span)", got, validSpanID)
	}
}

// TestContextFromTraceparent_MalformedReturnsInputCtx proves a malformed
// traceparent returns the input context UNCHANGED (no panic, no fabricated
// span context).
func TestContextFromTraceparent_MalformedReturnsInputCtx(t *testing.T) {
	in := context.Background()
	out := ContextFromTraceparent(in, "garbage", "")
	if trace.SpanContextFromContext(out).IsValid() {
		t.Errorf("malformed traceparent must not yield a valid active span context")
	}
}

// TestStartLinkedInboundSpan_AddsLink proves the started span carries a Link to
// the inbound orchestrator SpanContext (the achievable cross-process linkage).
func TestStartLinkedInboundSpan_AddsLink(t *testing.T) {
	var out []sdktrace.ReadOnlySpan
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(spanRecorder{out: &out}))
	// StartLinkedInboundSpan uses otel.Tracer(name); register tp globally.
	withGlobalTracerProvider(t, tp, func() {
		_, end := StartLinkedInboundSpan(context.Background(), "qgen_question", "qgen_question.inbound", validTraceparent, "")
		end()
	})

	if len(out) != 1 {
		t.Fatalf("expected 1 recorded span, got %d", len(out))
	}
	links := out[0].Links()
	if len(links) != 1 {
		t.Fatalf("expected 1 link to orchestrator span, got %d", len(links))
	}
	if got := links[0].SpanContext.TraceID().String(); got != validTraceID {
		t.Errorf("link TraceID=%q want %q", got, validTraceID)
	}
}

// TestStartLinkedInboundSpan_NoLinkOnMalformed proves a malformed traceparent
// still yields a usable span, just without a link.
func TestStartLinkedInboundSpan_NoLinkOnMalformed(t *testing.T) {
	var out []sdktrace.ReadOnlySpan
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(spanRecorder{out: &out}))
	withGlobalTracerProvider(t, tp, func() {
		_, end := StartLinkedInboundSpan(context.Background(), "qgen_critic", "qgen_critic.inbound", "garbage", "")
		end()
	})
	if len(out) != 1 {
		t.Fatalf("expected 1 recorded span, got %d", len(out))
	}
	if n := len(out[0].Links()); n != 0 {
		t.Errorf("expected 0 links on malformed traceparent, got %d", n)
	}
}
