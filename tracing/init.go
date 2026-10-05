// Package tracing wires the global OTel TracerProvider for ADK Go agents
// and continues inbound W3C trace context.
//
// Why this exists (debt close 2026-05-17 per FE-coord
// E2E-BE-AI-ASSIST-TRACE-EXPORT-PERM):
//
// The qgen_critic / qgen_question / familiar_companion / content_moderation /
// content_recommender / fog_orchestrator agent binaries emitted ZERO spans
// despite running successfully. The ADK Go SDK does NOT auto-wire a trace
// exporter — each agent main() must explicitly configure one. This package
// centralises the wiring so every agent main.go just calls
// `tracing.Init(ctx, serviceName)` instead of duplicating the OTel
// boilerplate.
//
// Cloud-neutral transport (2026-10-05): the Google Cloud Trace exporter was
// removed with the platform's Google Cloud exit. Init now delegates to
// chora-common/otel, which emits standard OTLP/gRPC to
// OTEL_EXPORTER_OTLP_ENDPOINT (the root compose runs an OTel Collector) and
// falls back to stdout in local dev, so startup never fails on trace wiring.
// The W3C TraceContext + Baggage global propagator is installed by the same
// call, so the orchestrator's traceparent — passed via session state / RPC
// metadata — is honoured and ADK-side spans continue the FE-originated trace
// tree.
//
// Usage in an agent main():
//
//	shutdown, err := tracing.Init(ctx, "qgen_critic")
//	if err != nil {
//	    log.Fatalf("tracing.Init: %v", err)
//	}
//	defer shutdown(context.Background())
//
// The TracerProvider is registered globally — downstream code calls
// `otel.Tracer("qgen_critic")` to get a scoped tracer. The agentengine
// launcher + the ADK runner pick up the global TracerProvider automatically
// for their built-in instrumentation hooks.
package tracing

import (
	"context"
	"errors"
	"os"
	"strings"

	chcommonotel "github.com/apollo-chora/chora-common/otel"
)

// Init wires the global TracerProvider + W3C propagator for the supplied
// serviceName and returns a shutdown func the caller MUST defer to flush any
// pending spans on exit.
//
// serviceName MUST be non-empty (the canonical OTel service.name attribute).
// service.version is read from CHORA_SERVICE_VERSION env (optional, default
// "dev") and stamped as the OTLP service.version resource attribute.
//
// Resource attributes follow the platform-wide convention from
// chora-common/otel: service.name, service.version, service.namespace=chora,
// deployment.environment (DEPLOYMENT_ENVIRONMENT env, default "dev").
//
// Sampler: chora-common/otel defaults to always-on (every span is exported)
// and honours OTEL_TRACES_SAMPLER / OTEL_TRACES_SAMPLER_ARG for cost-aware
// sampling.
func Init(ctx context.Context, serviceName string) (func(context.Context) error, error) {
	if strings.TrimSpace(serviceName) == "" {
		return nil, errors.New("tracing.Init: serviceName must be non-empty")
	}

	version := strings.TrimSpace(os.Getenv("CHORA_SERVICE_VERSION"))
	if version == "" {
		version = "dev"
	}
	return chcommonotel.Init(ctx, serviceName, version)
}
