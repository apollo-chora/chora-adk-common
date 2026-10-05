package labeledgemini

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// tracer is package-scoped so callers don't need to plumb it. Uses the
// global TracerProvider (which ADK Go's telemetry package installs from
// the launcher's Setup func — see google.golang.org/adk/telemetry).
//
// The tracer name follows the OpenInference convention for LLM spans
// — see ai-observability-cloud-trace skill.
var tracer = otel.Tracer("chora-adk-common/labeledgemini")

// ---------- ports ----------

// LabeledLLM is the inbound port: ADK Go agent code depends on this
// interface, not on a concrete model client. Two impls today:
//
//   - ChoraLabeledGemini wraps a concrete inner LabeledLLM
//     (per ADR-146 §5 — the canonical Chora extension).
//   - FakeLabeledLLM is the deterministic test adapter.
//
// There is deliberately no direct-Vertex adapter here any more. The
// VertexGemini adapter that used to sit in vertex.go called Vertex AI
// straight, off the chokepoint, and had zero callers; it was deleted with
// G1'-6 (2026-08-07). Production traffic reaches a model through
// chora-model-gateway (ADR-163/177), which is where Armor screening,
// metering and the token ledger live. See modelgatewayclient.
//
// Hexagonal: do NOT import google.golang.org/genai in this interface —
// keep the port narrow + vendor-agnostic.
type LabeledLLM interface {
	// Model returns the underlying model name (e.g., "gemini-2.5-flash").
	// Surfaced in trace attributes + event payloads.
	Model() string

	// Generate runs a single prompt → response. Implementations
	// MUST set the cost-attribution labels on the outgoing request,
	// route LoRA adapters per the AdapterPath, and capture usage
	// in the response.
	Generate(ctx context.Context, req GenerateRequest) (GenerateResponse, error)
}

// GenerateRequest is the inbound LabeledLLM request shape. Keeps the
// minimum needed for the Chora wrapper's invariants — label stamping +
// LoRA routing + cost event emission. Inner adapters layer their own
// vendor-specific request shape underneath (e.g., genai.GenerateContentRequest).
type GenerateRequest struct {
	// Required attribution — Chora invariants.
	TenantID   string // chora_tenant_id cost-attribution label
	UserGCID   string // chora_gcid cost-attribution label
	CrewName   string // chora_crew_name cost-attribution label
	AgentRole  string // chora_agent_role cost-attribution label
	WorkflowID string // chora_workflow_id cost-attribution label (optional but recommended)

	// Prompt is the user-facing prompt text. Production wiring layers
	// system instructions + RAG chunks underneath this — that shape
	// stays in the inner adapter, not the port, so the port doesn't
	// drift with the vendor SDK.
	Prompt string

	// AdapterPath, when non-empty, instructs the inner adapter to
	// route to the per-tenant LoRA-adapted Gemma endpoint.
	// Resolved by ChoraLabeledGemini via LoRARouter; the test adapter
	// just records what it received.
	AdapterPath string

	// Labels is the cost-attribution labels map the inner adapter
	// stamps onto the outgoing call (e.g., genai.Config.Labels).
	// ChoraLabeledGemini populates this from the request fields +
	// caller-supplied ExtraLabels before passing through.
	Labels map[string]string

	// ExtraLabels lets callers attach crew-specific labels (e.g.,
	// `familiar_id`, `familiar_specialization`). Merged WITH the
	// Chora invariants — Chora labels win on key collision.
	ExtraLabels map[string]string
}

// GenerateResponse is the inbound LabeledLLM response shape.
type GenerateResponse struct {
	Text         string
	InputTokens  int
	OutputTokens int
	ModelVersion string // surfaced from vendor SDK if available
}

// ---------- ChoraLabeledGemini ----------

// Config wires ChoraLabeledGemini for a specific crew + deploy target.
type Config struct {
	// Inner is the concrete LabeledLLM (a gateway-backed adapter in prod,
	// FakeLabeledLLM in tests). REQUIRED.
	Inner LabeledLLM

	// LoRARouter resolves per-tenant adapters. Optional — defaults
	// to NoopLoRARouter (every tenant gets the base model).
	LoRARouter LoRARouter

	// ChoraEnv is the deploy stage (`dev` | `staging` | `prod`).
	// Stamped as the `chora_env` cost-attribution label. Pull from
	// manaplugin.ChoraEnv() at construction time.
	ChoraEnv string

	// EventSink receives `model.invoked.v1` events for cost
	// attribution. Optional — defaults to NullEventSink (events
	// discarded; useful for unit tests that don't care).
	EventSink EventSink

	// Now returns the current time. Optional — defaults to time.Now.
	// Injection seam for latency assertions in tests.
	Now func() time.Time
}

// ChoraLabeledGemini is the canonical Chora "ChoraLabeledGemini" wrapper
// referenced in ADR-146 §5. Stamps cost-attribution labels per request,
// routes per-tenant LoRA adapters, emits cost-tracking events, and
// wraps every call in an OTel span with OpenInference semantic
// conventions (`gen_ai.system=vertex_ai`, `gen_ai.request.model`,
// `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`).
type ChoraLabeledGemini struct {
	inner    LabeledLLM
	router   LoRARouter
	choraEnv string
	sink     EventSink
	now      func() time.Time
}

// New constructs a ChoraLabeledGemini. Returns an error if Inner is nil.
// Other Config fields default to safe values (NoopLoRARouter / NullEventSink).
func New(cfg Config) (*ChoraLabeledGemini, error) {
	if cfg.Inner == nil {
		return nil, errors.New("labeledgemini: Inner LabeledLLM required")
	}
	if cfg.LoRARouter == nil {
		cfg.LoRARouter = NewNoopLoRARouter()
	}
	if cfg.EventSink == nil {
		cfg.EventSink = NewNullEventSink()
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	return &ChoraLabeledGemini{
		inner:    cfg.Inner,
		router:   cfg.LoRARouter,
		choraEnv: cfg.ChoraEnv,
		sink:     cfg.EventSink,
		now:      cfg.Now,
	}, nil
}

// Model returns the inner LLM's model name.
func (g *ChoraLabeledGemini) Model() string {
	return g.inner.Model()
}

// Generate runs the request through the Chora wrapper:
//
//  1. Validate Chora invariants (tenant_id + user_gcid + crew_name required).
//  2. Resolve per-tenant LoRA adapter (best-effort — fall back to
//     base model on router error per ADR-146 Resilience).
//  3. Compose cost-attribution labels (chora.* + caller ExtraLabels).
//  4. Open OTel span with OpenInference attributes.
//  5. Delegate to inner LabeledLLM.
//  6. Emit `model.invoked.v1` event with usage + latency.
//  7. Close span (status set per inner result).
func (g *ChoraLabeledGemini) Generate(
	ctx context.Context,
	req GenerateRequest,
) (GenerateResponse, error) {
	if err := validateRequest(req); err != nil {
		return GenerateResponse{}, err
	}

	// Step 2: LoRA resolution (fail-open).
	adapter, adapterErr := g.router.ResolveAdapter(ctx, req.TenantID)
	if adapterErr != nil {
		// Fail-open: blocking the call on an adapter lookup is worse
		// than serving the base model. Span event records the fallback.
		adapter = AdapterRef{}
	}

	// Step 3: compose labels — Chora invariants win on key collision.
	labels := mergeLabels(req.ExtraLabels, map[string]string{
		"chora_tenant_id":   req.TenantID,
		"chora_gcid":        req.UserGCID,
		"chora_crew_name":   req.CrewName,
		"chora_agent_role":  req.AgentRole,
		"chora_workflow_id": req.WorkflowID,
		"chora_env":         g.choraEnv,
	})
	if adapter.AdapterPath != "" {
		labels["chora_lora_adapter"] = adapter.AdapterPath
	}

	// Step 4: OTel span — OpenInference conventions.
	ctx, span := tracer.Start(ctx, "labeledgemini.generate",
		trace.WithAttributes(
			attribute.String("gen_ai.system", "vertex_ai"),
			attribute.String("gen_ai.request.model", g.inner.Model()),
			attribute.String("chora.tenant_id", req.TenantID),
			attribute.String("chora.crew_name", req.CrewName),
			attribute.String("chora.agent_role", req.AgentRole),
			attribute.String("chora.workflow_id", req.WorkflowID),
			attribute.String("chora.env", g.choraEnv),
		),
	)
	defer span.End()
	if adapter.AdapterPath != "" {
		span.SetAttributes(
			attribute.String("chora.lora_adapter", adapter.AdapterPath),
			attribute.Int("chora.lora_version", adapter.Version),
		)
	}
	if adapterErr != nil {
		span.AddEvent("lora_router_fallback",
			trace.WithAttributes(attribute.String("error", adapterErr.Error())),
		)
	}

	// Step 5: inner call.
	innerReq := req
	innerReq.AdapterPath = adapter.AdapterPath
	innerReq.Labels = labels
	start := g.now()
	resp, err := g.inner.Generate(ctx, innerReq)
	latency := g.now().Sub(start)

	// Step 6: emit cost event (best-effort; errors logged not propagated
	// — emitting MUST NOT block the response per outbox-style design;
	// the production EventSink writes to a durable outbox.).
	emitErr := g.sink.Emit(ctx, ModelInvokedEvent{
		EventType:    "model.invoked.v1",
		TenantID:     req.TenantID,
		UserGCID:     req.UserGCID,
		CrewName:     req.CrewName,
		AgentRole:    req.AgentRole,
		WorkflowID:   req.WorkflowID,
		Model:        g.inner.Model(),
		AdapterPath:  adapter.AdapterPath,
		AdapterVer:   adapter.Version,
		InputTokens:  resp.InputTokens,
		OutputTokens: resp.OutputTokens,
		LatencyMS:    latency.Milliseconds(),
		ErrorCode:    errorCode(err),
		OccurredAt:   start,
		ChoraEnv:     g.choraEnv,
	})
	if emitErr != nil {
		span.AddEvent("event_sink_emit_failed",
			trace.WithAttributes(attribute.String("error", emitErr.Error())),
		)
	}

	// Step 7: span status + usage attrs.
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
	} else {
		span.SetAttributes(
			attribute.Int("gen_ai.usage.input_tokens", resp.InputTokens),
			attribute.Int("gen_ai.usage.output_tokens", resp.OutputTokens),
		)
	}
	span.SetAttributes(attribute.Int64("chora.latency_ms", latency.Milliseconds()))

	return resp, err
}

func validateRequest(req GenerateRequest) error {
	if req.TenantID == "" {
		return errors.New("labeledgemini: TenantID required (chora_tenant_id label)")
	}
	if req.UserGCID == "" {
		return errors.New("labeledgemini: UserGCID required (chora_gcid label)")
	}
	if req.CrewName == "" {
		return errors.New("labeledgemini: CrewName required (chora_crew_name label)")
	}
	return nil
}

// mergeLabels merges base + overrides, with overrides winning on key
// collision. Returns a new non-nil map. Nil-safe.
func mergeLabels(base, overrides map[string]string) map[string]string {
	out := make(map[string]string, len(base)+len(overrides))
	for k, v := range base {
		out[k] = v
	}
	for k, v := range overrides {
		if v != "" {
			out[k] = v
		}
	}
	return out
}

func errorCode(err error) string {
	if err == nil {
		return ""
	}
	// Production: classify into a stable code (e.g., "vertex.unavailable",
	// "vertex.quota_exhausted"). POC: stringify error type for now.
	return fmt.Sprintf("%T", err)
}

// ---------- ModelInvokedEvent + EventSink ----------

// ModelInvokedEvent is the payload emitted on every Generate call. In
// production the EventSink writes to an outbox table that publishes to
// `chora.ai_kernel.token-usage.recorded.v1` per ai-cost-tracking skill +
// ADR-146 §IMDA D3 cost+monitoring evidence stream.
type ModelInvokedEvent struct {
	EventType    string // "model.invoked.v1"
	TenantID     string
	UserGCID     string
	CrewName     string
	AgentRole    string
	WorkflowID   string
	Model        string
	AdapterPath  string // empty for base model
	AdapterVer   int
	InputTokens  int
	OutputTokens int
	LatencyMS    int64
	ErrorCode    string // empty on success
	OccurredAt   time.Time
	ChoraEnv     string
}

// EventSink is the outbound port for cost event emission. Production
// adapter writes to a outbox table; POC tests use
// RecordingEventSink; deploys that don't care use NullEventSink.
type EventSink interface {
	Emit(ctx context.Context, ev ModelInvokedEvent) error
}

// RecordingEventSink is the in-memory test adapter for EventSink.
type RecordingEventSink struct {
	mu     sync.Mutex
	events []ModelInvokedEvent
}

// NewRecordingEventSink constructs an empty recording sink.
func NewRecordingEventSink() *RecordingEventSink { return &RecordingEventSink{} }

// Emit appends to the in-memory log. Always nil error.
func (s *RecordingEventSink) Emit(_ context.Context, ev ModelInvokedEvent) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, ev)
	return nil
}

// Events returns a copy of the recorded events.
func (s *RecordingEventSink) Events() []ModelInvokedEvent {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]ModelInvokedEvent, len(s.events))
	copy(out, s.events)
	return out
}

// NullEventSink discards every event. Useful as a default when callers
// don't care about cost telemetry (rare in production).
type NullEventSink struct{}

// NewNullEventSink constructs the null sink.
func NewNullEventSink() NullEventSink { return NullEventSink{} }

// Emit is a no-op.
func (NullEventSink) Emit(_ context.Context, _ ModelInvokedEvent) error { return nil }

// ---------- FakeLabeledLLM ----------

// FakeLabeledLLM is the deterministic test adapter for LabeledLLM.
// Records every Generate invocation in order.
type FakeLabeledLLM struct {
	mu        sync.Mutex
	modelName string
	responses []GenerateResponse
	errors    []error
	calls     []FakeCall
}

// FakeCall captures one Generate invocation.
type FakeCall struct {
	TenantID    string
	UserGCID    string
	CrewName    string
	AgentRole   string
	WorkflowID  string
	Prompt      string
	AdapterPath string
	Labels      map[string]string
}

// NewFakeLabeledLLM constructs a fake with the given model name.
func NewFakeLabeledLLM(modelName string) *FakeLabeledLLM {
	return &FakeLabeledLLM{modelName: modelName}
}

// Model implements LabeledLLM.
func (f *FakeLabeledLLM) Model() string { return f.modelName }

// QueueResponse adds a canned response to the FIFO. Empty queue =>
// Generate returns a zero response.
func (f *FakeLabeledLLM) QueueResponse(resp GenerateResponse) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.responses = append(f.responses, resp)
}

// QueueError makes the NEXT Generate call return err (instead of any
// queued response).
func (f *FakeLabeledLLM) QueueError(err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.errors = append(f.errors, err)
}

// Generate implements LabeledLLM. Records the call, then returns the
// next queued error / response (errors take priority).
func (f *FakeLabeledLLM) Generate(_ context.Context, req GenerateRequest) (GenerateResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, FakeCall{
		TenantID:    req.TenantID,
		UserGCID:    req.UserGCID,
		CrewName:    req.CrewName,
		AgentRole:   req.AgentRole,
		WorkflowID:  req.WorkflowID,
		Prompt:      req.Prompt,
		AdapterPath: req.AdapterPath,
		Labels:      copyLabels(req.Labels),
	})
	if len(f.errors) > 0 {
		err := f.errors[0]
		f.errors = f.errors[1:]
		return GenerateResponse{}, err
	}
	if len(f.responses) > 0 {
		resp := f.responses[0]
		f.responses = f.responses[1:]
		return resp, nil
	}
	return GenerateResponse{}, nil
}

// Calls returns a copy of the recorded calls.
func (f *FakeLabeledLLM) Calls() []FakeCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]FakeCall, len(f.calls))
	copy(out, f.calls)
	return out
}

func copyLabels(in map[string]string) map[string]string {
	if in == nil {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
