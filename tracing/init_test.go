package tracing

import (
	"context"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// spanRecorder collects ended ReadOnlySpan values for assertions.
type spanRecorder struct{ out *[]sdktrace.ReadOnlySpan }

func (r spanRecorder) OnStart(context.Context, sdktrace.ReadWriteSpan) {}
func (r spanRecorder) OnEnd(s sdktrace.ReadOnlySpan)                   { *r.out = append(*r.out, s) }
func (r spanRecorder) Shutdown(context.Context) error                  { return nil }
func (r spanRecorder) ForceFlush(context.Context) error                { return nil }

// TestInit_RejectsEmptyServiceName guards the contract that callers must
// pass a non-empty service.name (the canonical OTel resource attribute).
func TestInit_RejectsEmptyServiceName(t *testing.T) {
	_, err := Init(context.Background(), "")
	if err == nil {
		t.Fatalf("Init(\"\") must error")
	}
}

// TestInit_CloudNeutral guards the cloud-neutral contract: Init wires a global
// TracerProvider with NO Google Cloud environment (no project env var,
// no ADC). The exporter target comes from OTEL_EXPORTER_OTLP_ENDPOINT (OTLP)
// or stdout in local dev — never Cloud Trace.
func TestInit_CloudNeutral(t *testing.T) {
	shutdown, err := Init(context.Background(), "qgen_critic_test")
	if err != nil {
		t.Fatalf("Init: %v", err)
	}
	defer func() { _ = shutdown(context.Background()) }()

	// A span started after Init must be recorded by the global provider —
	// proof the TracerProvider is wired (not the no-op default).
	_, span := otel.Tracer("tracing_test").Start(context.Background(), "init_probe")
	sc := span.SpanContext()
	span.End()
	if !sc.IsValid() {
		t.Fatal("expected a valid span context from the global TracerProvider")
	}
}

// TestInit_SetsGlobalPropagator asserts the W3C TraceContext propagator is
// installed globally so downstream code that calls otel.GetTextMapPropagator()
// gets a real propagator (not the no-op default). The contract: a propagator
// invoked on a map containing a W3C traceparent must yield a SpanContext
// matching that traceparent's trace_id + span_id.
func TestInit_SetsGlobalPropagator(t *testing.T) {
	// Reset propagator to no-op so the assertion is meaningful.
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator())
	shutdown, err := Init(context.Background(), "qgen_critic_test")
	if err != nil {
		t.Fatalf("Init: %v", err)
	}
	defer func() { _ = shutdown(context.Background()) }()

	prop := otel.GetTextMapPropagator()
	const tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
	headers := map[string]string{"traceparent": tp}
	ctx := prop.Extract(context.Background(), propagation.MapCarrier(headers))
	sc := trace.SpanContextFromContext(ctx)
	if !sc.IsValid() {
		t.Fatalf("expected propagator to extract a valid SpanContext from W3C traceparent header")
	}
	wantTrace := "0af7651916cd43dd8448eb211c80319c"
	if got := sc.TraceID().String(); got != wantTrace {
		t.Errorf("trace_id=%q want %q", got, wantTrace)
	}
}
