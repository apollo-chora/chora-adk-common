# chora-adk-common

Shared library for the Chora ADK Go agent crews (Familiar, Question
Generation, and the moderation / recommender / evaluation crews). Every crew
imports this module for the per-user mana gate, Chora-context label injection,
the `TenantManaPool` client, the model-gateway LLM adapter, Agent Card
publishing, and the event-driven dispatch subscriber.

Module path: `github.com/apollo-chora/chora-adk-common`.

The library is cloud-neutral: NATS JetStream for events (via
`chora-common/eventbus`), standard OTLP for traces (via `chora-common/otel`),
and env-backed secrets. No cloud account or managed service is required.

## Packages

| Package | Purpose |
|---|---|
| `agentcard/` | A2A v1.0 Agent Card authoring (`Builder`) + the `/.well-known/agent-card` HTTP handler. |
| `agentdispatch/` | ADR-253/254 event-driven dispatch: a NATS JetStream subscriber inside the agent binary that runs the ADK agent in-process and publishes a completion. |
| `ahamomentplugin/` | ADR-149 Stage-3 Aha-moment preview override (one-shot 24h Stage-6 lift). |
| `groundingplugin/` | EPIC-1a batch source-material grounding — injects the uploaded object-store URI as a `FileData` part. |
| `growthstageplugin/` | ADR-149 growth-stage capability gating (`growth.allowed_tools` / `growth.memory_mode`), loaded from an embedded `config.yaml`. |
| `instancedispatch/` | ADR-147 §7 session-time per-instance config: `Resolver` + tool filter + `InstructionProvider`. |
| `labeledgemini/` | `ChoraLabeledGemini` wrapper — per-request cost-attribution labels, per-tenant adapter routing, cost-event emission, OTel spans. |
| `manaplugin/` | ADR-142 per-user mana gate + Chora-context label injection. The only Chora-side runtime code on the model call path. |
| `modelgatewayclient/` | `chora-model-gateway` gRPC `Invoke` adapted to the ADK `model.LLM` interface, plus the image invoker and the tenant-propagation plugin. |
| `promptstamping/` | ADR-197 M-A prompt-version / content-hash / condition span stamping (behaviour-neutral). |
| `tenancy/` | `ManaClient` interface + `TenantManaPool` adapter. |
| `terminationplugin/` | Structured `AgentTerminated` emission for every agent run lifecycle. |
| `tracing/` | Global OTel `TracerProvider` wiring + inbound W3C `traceparent` continuation. |

## Usage from a crew

```go
import (
    "github.com/apollo-chora/chora-adk-common/manaplugin"
    "github.com/apollo-chora/chora-adk-common/tenancy"
)

func main() {
    mana, _ := tenancy.New()
    manaP, _ := manaplugin.New(manaplugin.Config{
        Mana:     mana,
        CrewKind: "familiar",   // or "qgen", "delivery_rostering", etc.
    })
    // ... wire into launcher.Config{ PluginConfig: ... }
}
```

## Multi-crew safety

`manaplugin.New(...)` REFUSES to build a plugin without `CrewKind` set. This is
non-negotiable — without it, cost attribution loses crew identity, breaking the
per-crew cost dashboards and the IMDA D3 evidence stream. `growthstageplugin`,
`ahamomentplugin`, and `modelgatewayclient.NewTenantPropagationPlugin` apply the
same rule.

## Transport

- **Events** — NATS JetStream via `chora-common/eventbus`. The dispatch topics
  (`chora.ai_kernel.agent_dispatch.<role>_{requested,completed}.v1`) are valid
  NATS subjects; the canonical event envelope rides as NATS headers.
- **Traces** — standard OTLP/gRPC via `chora-common/otel`; stdout in local dev
  when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset.
- **Model calls** — gRPC to `chora-model-gateway`; TLS + `CHORA_GATEWAY_TOKEN`
  in production, plaintext when `Insecure` is set for local dev.

## Configuration

| Variable | Purpose | Local default |
| --- | --- | --- |
| `NATS_URL` | NATS JetStream event bus (dispatch subscriber) | unset |
| `AGENT_DISPATCH_ENABLED` | Opt in to the dispatch subscriber | `false` |
| `AGENT_DISPATCH_SUBSCRIPTION` | Consumer name override | derived from service + role |
| `AGENT_DISPATCH_MAX_DELIVERY_ATTEMPTS` | Redelivery ceiling (must match the consumer) | `5` |
| `AGENT_HEALTH_PORT` | Health/readiness port for a subscriber-only agent | `8080` |
| `CHORA_GATEWAY_TOKEN` | Static bearer token for the model gateway | unset |
| `CHORA_GATEWAY_INSECURE` | Plaintext gRPC to a local gateway (dev only) | unset |
| `TENANCY_GRPC_ENDPOINT` | `TenantManaPool` endpoint (`stub://…` for the in-memory stub) | unset |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP/gRPC trace endpoint | stdout |
| `CHORA_SERVICE_VERSION` | Stamped as the OTLP `service.version` attribute | `dev` |

This module is a library — it has no service binary of its own and no port.
A crew that runs a subscriber-only agent binds the `agentdispatch` health
server on `AGENT_HEALTH_PORT` (default `8080`); it needs no database.

## Build and test

```sh
go build ./...
go vet ./...
go test ./...
```

The suite is hermetic — no broker, database, or network is required.
