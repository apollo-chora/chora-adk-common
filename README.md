# chora-adk-common

## About

`chora-adk-common` is a shared Go library for Chora agents built with the Google ADK. It provides common packages for A2A Agent Cards, event-driven agent dispatch, mana checks and cost attribution, model-gateway access, per-instance configuration, growth-stage capability gating, source-material grounding, prompt evidence, termination events, tenancy, and OpenTelemetry tracing. The module is consumed by agent services rather than run as a standalone service.

## Quick start

Requires Go 1.26.6, as declared by `go.mod`.

Clone the repository:

```sh
git clone https://github.com/apollo-chora/chora-adk-common.git
cd chora-adk-common
```

Build and test the library:

```sh
go build ./...
go vet ./...
go test ./...
```

The test suite is hermetic and does not require a broker, database, or network connection.

To use the module from another Go project:

```sh
go get github.com/apollo-chora/chora-adk-common
```

## Usage

Import the package that provides the capability needed by the agent. The main packages are:

| Package | Purpose |
| --- | --- |
| `agentcard` | Build an A2A v1.0 Agent Card and serve it at `/.well-known/agent-card`. |
| `agentdispatch` | Consume A2A agent-dispatch events from NATS JetStream, run the ADK agent in-process, and publish a completion. |
| `ahamomentplugin` | Apply the Stage-3, time-bounded Aha-moment capability preview. |
| `groundingplugin` | Add source material from an object-store URI in session state as an ADK `FileData` part. |
| `growthstageplugin` | Resolve the Familiar growth stage into allowed tools and memory mode from the embedded `config.yaml`. |
| `instancedispatch` | Resolve per-session instance configuration and filter the model tool surface. |
| `labeledgemini` | Wrap an LLM with Chora cost-attribution labels, optional per-tenant LoRA routing, events, and tracing. |
| `manaplugin` | Enforce the per-user mana gate and add Chora context to the ADK run. |
| `modelgatewayclient` | Adapt the Chora model-gateway gRPC `Invoke` API to the ADK `model.LLM` interface. |
| `promptstamping` | Stamp prompt version, rendered-prompt SHA-256, and composition conditions onto the active OTel span without changing the prompt. |
| `tenancy` | Define the `ManaClient` interface and create the tenant mana client or development stub. |
| `terminationplugin` | Emit a structured `AgentTerminated` event for an agent run. |
| `tracing` | Configure the global OTel provider and continue inbound W3C trace context. |

### Agent Card

Build a card with the required name, version, and AGID, then register its handler on a standard `http.ServeMux`:

```go
card, err := agentcard.NewBuilder().
    WithName("familiar_companion").
    WithVersion("1.0.0").
    WithAGID("01970000-0000-7000-b000-000000000001").
    Build()
if err != nil {
    return err
}

agentcard.RegisterMux(mux, card)
```

The handler serves `GET /.well-known/agent-card` with JSON and returns `405` for other methods.

### Mana plugin

Every crew using the mana gate supplies both a `ManaClient` and a non-empty `CrewKind`:

```go
mana, err := tenancy.New()
if err != nil {
    return err
}

manaPlugin, err := manaplugin.New(manaplugin.Config{
    Mana:     mana,
    CrewKind: "familiar",
})
if err != nil {
    return err
}
```

`manaplugin.New` rejects configurations without a mana client or crew identifier. The plugin expects `tenant_id` and `user_gcid` in ADK session state before a model call.

### Instance dispatch

Provide a resolver and the session-state key that identifies the instance:

```go
cfg := instancedispatch.Config{
    Resolver: resolver,
    StateKey: "familiar_id",
}

dispatchPlugin, err := instancedispatch.New(cfg)
if err != nil {
    return err
}

instructionProvider, err := instancedispatch.NewInstructionProvider(cfg)
if err != nil {
    return err
}
```

The plugin resolves the instance before a run and filters the model's available tools on each model turn. The instruction provider supplies the resolved per-instance system instruction.

### Model gateway client

The model gateway adapter is configured explicitly:

```go
client, err := modelgatewayclient.New(modelgatewayclient.Config{
    Endpoint:       "gateway.example:443",
    LogicalModelID: "gemini-2.5-flash",
    AgentID:        "familiar_companion",
    Surface:        "companion_chat",
    TenantID:       tenantID,
    GCID:           userGCID,
})
if err != nil {
    return err
}
```

The client sends unary gRPC `Invoke` requests to `chora-model-gateway`. TLS is used by default; setting `Insecure` switches to plaintext gRPC for local development.

### Agent dispatch

The dispatch package uses NATS JetStream. Request and completion subjects are derived from the agent role:

```go
requestTopic := agentdispatch.RequestTopic("qgen_question")
completionTopic := agentdispatch.CompletionTopic("qgen_question")
```

The subscriber can run alongside the normal ADK launcher with `Serve`, or as a subscriber-only process with `RunSubscriberOnly`.

### Growth-stage and grounding plugins

`growthstageplugin` reads `growth_stage`, `species`, and optional visible KG neighbors from session state and writes the resolved growth capability state back to the session. Its default seven-stage table is embedded from `growthstageplugin/config.yaml`.

`groundingplugin.New(crewKind)` creates a `BeforeModelCallback` that reads `source_blob_uri` and `source_mime_type` from session state. When a source URI is present, it appends a `FileData` part to the last user content.

### Environment variables

The library has no service-wide configuration file. Environment variables are read only by the packages that need them:

| Variable | Used for | Default |
| --- | --- | --- |
| `TENANCY_GRPC_ENDPOINT` | `tenancy.New()` client selection. A `stub://...` value selects the in-memory stub. | unset |
| `CHORA_GATEWAY_TOKEN` | Bearer token for model-gateway gRPC calls. | unset |
| `CHORA_GATEWAY_INSECURE` | Local plaintext model-gateway connection when enabled by the caller's configuration. | unset |
| `CHORA_GATEWAY_TENANT_ID` | Fallback tenant identity for the model-gateway client. | unset |
| `CHORA_GATEWAY_GCID` | Fallback GCID/agent identity for the model-gateway client. | unset |
| `NATS_URL` | NATS JetStream endpoint for `agentdispatch`. | unset |
| `AGENT_DISPATCH_ENABLED` | Enables subscriber-only dispatch startup. | `false` |
| `AGENT_DISPATCH_SUBSCRIPTION` | Overrides the dispatch consumer name. | derived from service and role |
| `AGENT_DISPATCH_MAX_DELIVERY_ATTEMPTS` | Delivery-attempt ceiling used by dispatch handling. | `5` |
| `AGENT_HEALTH_PORT` | Health/readiness port for subscriber-only mode. | `8080` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP/gRPC trace exporter endpoint. | stdout via `chora-common/otel` |
| `CHORA_SERVICE_VERSION` | OTel `service.version` used by `tracing.Init`. | `dev` |

## Development

The repository is a Go module with package-level tests. There is no application binary and no runtime port owned by the library.

Run the checks used by CI:

```sh
gofmt -l .
go mod tidy
go vet ./...
go test ./...
```

CI also verifies that `go mod tidy` produces no changes to `go.mod` or `go.sum`.

The main source tree is organized by package:

```text
agentcard/             A2A Agent Card types, builder, and HTTP handler
agentdispatch/         Event-driven agent subscriber and execution flow
ahamomentplugin/       Stage-3 Aha-moment plugin
groundingplugin/       Source-material grounding plugin
growthstageplugin/     Growth-stage capability configuration and plugin
instancedispatch/      Per-session instance resolver and tool filtering
labeledgemini/         Cost labels, adapter routing, events, and tracing
manaplugin/            Per-user mana gate and Chora context labels
modelgatewayclient/    ADK LLM adapter for chora-model-gateway
promptstamping/        Prompt-version and content-hash span stamping
tenancy/               Tenant mana client interface and stub
terminationplugin/     Agent termination event plugin
tracing/               Global OTel setup and W3C trace continuation
```

`growthstageplugin/config.yaml` is embedded into the package at build time. The Dockerfile is also a library-build artifact: its builder stage runs `go build ./...`, `go vet ./...`, and `go test ./...`, while the final image provides the source tree and Go module cache for downstream consumers.
