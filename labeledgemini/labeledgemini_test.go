package labeledgemini

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// ---------- LoRARouter / InMemoryLoRARouter ----------

func TestInMemoryLoRARouter_returnsRegisteredAdapter(t *testing.T) {
	r := NewInMemoryLoRARouter()
	r.Set("acme", AdapterRef{AdapterPath: "tenant-acme-v3", Version: 3})

	ref, err := r.ResolveAdapter(context.Background(), "acme")
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if ref.AdapterPath != "tenant-acme-v3" {
		t.Errorf("want adapter path tenant-acme-v3; got %q", ref.AdapterPath)
	}
	if ref.Version != 3 {
		t.Errorf("want version 3; got %d", ref.Version)
	}
}

func TestInMemoryLoRARouter_returnsBaseModelForUnknownTenant(t *testing.T) {
	r := NewInMemoryLoRARouter()
	r.Set("acme", AdapterRef{AdapterPath: "tenant-acme-v3", Version: 3})

	ref, err := r.ResolveAdapter(context.Background(), "globex")
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if ref.AdapterPath != "" {
		t.Errorf("unknown tenant should resolve to base model (empty); got %q", ref.AdapterPath)
	}
}

func TestInMemoryLoRARouter_rejectsEmptyTenantID(t *testing.T) {
	r := NewInMemoryLoRARouter()
	_, err := r.ResolveAdapter(context.Background(), "")
	if err == nil {
		t.Fatal("want error for empty tenantID; got nil")
	}
}

func TestInMemoryLoRARouter_surfacesInjectedError(t *testing.T) {
	r := NewInMemoryLoRARouter()
	wantErr := errors.New("cloud sql timeout")
	r.SetError(wantErr)
	_, err := r.ResolveAdapter(context.Background(), "acme")
	if !errors.Is(err, wantErr) {
		t.Errorf("want injected error; got %v", err)
	}
}

func TestNoopLoRARouter_alwaysBaseModel(t *testing.T) {
	r := NewNoopLoRARouter()
	ref, err := r.ResolveAdapter(context.Background(), "any-tenant")
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if ref.AdapterPath != "" {
		t.Errorf("noop should always return base model; got %q", ref.AdapterPath)
	}
}

// ---------- LabeledLLM / FakeLabeledLLM ----------

func TestFakeLabeledLLM_recordsCalls(t *testing.T) {
	f := NewFakeLabeledLLM("gemini-test")
	f.QueueResponse(GenerateResponse{
		Text:         "hello",
		InputTokens:  10,
		OutputTokens: 20,
	})

	resp, err := f.Generate(context.Background(), GenerateRequest{
		TenantID:  "acme",
		UserGCID:  "u-1",
		CrewName:  "familiar",
		AgentRole: "companion",
		Prompt:    "hi",
	})
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if resp.Text != "hello" {
		t.Errorf("want hello; got %q", resp.Text)
	}
	if len(f.Calls()) != 1 {
		t.Fatalf("want 1 recorded call; got %d", len(f.Calls()))
	}
	call := f.Calls()[0]
	if call.TenantID != "acme" {
		t.Errorf("want tenant acme; got %q", call.TenantID)
	}
}

func TestFakeLabeledLLM_surfacesQueuedError(t *testing.T) {
	f := NewFakeLabeledLLM("gemini-test")
	wantErr := errors.New("vertex 500")
	f.QueueError(wantErr)

	_, err := f.Generate(context.Background(), GenerateRequest{
		TenantID: "acme", UserGCID: "u", CrewName: "c", AgentRole: "r", Prompt: "x",
	})
	if !errors.Is(err, wantErr) {
		t.Errorf("want queued error; got %v", err)
	}
}

func TestFakeLabeledLLM_satisfiesLabeledLLMInterface(t *testing.T) {
	var _ LabeledLLM = NewFakeLabeledLLM("gemini-test")
}

// ---------- ChoraLabeledGemini composition wrapper ----------

func TestChoraLabeledGemini_stampsCloudBillingLabels(t *testing.T) {
	fake := NewFakeLabeledLLM("gemini-2.5-flash")
	fake.QueueResponse(GenerateResponse{Text: "ok", InputTokens: 5, OutputTokens: 7})

	cfg := Config{
		Inner:      fake,
		LoRARouter: NewNoopLoRARouter(),
		ChoraEnv:   "dev",
	}
	llm, err := New(cfg)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}

	_, err = llm.Generate(context.Background(), GenerateRequest{
		TenantID:   "acme",
		UserGCID:   "01957c8c-aaaa-7000-bbbb-cccccccccccc",
		CrewName:   "familiar",
		AgentRole:  "companion",
		WorkflowID: "wf-1",
		Prompt:     "hello",
	})
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}

	call := fake.Calls()[0]
	if call.Labels["chora_tenant_id"] != "acme" {
		t.Errorf("want chora_tenant_id label; got %q", call.Labels["chora_tenant_id"])
	}
	if call.Labels["chora_crew_name"] != "familiar" {
		t.Errorf("want chora_crew_name label; got %q", call.Labels["chora_crew_name"])
	}
	if call.Labels["chora_agent_role"] != "companion" {
		t.Errorf("want chora_agent_role label; got %q", call.Labels["chora_agent_role"])
	}
	if call.Labels["chora_workflow_id"] != "wf-1" {
		t.Errorf("want chora_workflow_id label; got %q", call.Labels["chora_workflow_id"])
	}
	if call.Labels["chora_env"] != "dev" {
		t.Errorf("want chora_env=dev label; got %q", call.Labels["chora_env"])
	}
	// GCID label MUST be present but the full UUID must NOT appear
	// verbatim in error messages — privacy invariant. We just check
	// the label is set (Cloud Billing scope is controlled access).
	if call.Labels["chora_gcid"] == "" {
		t.Errorf("want chora_gcid label set; got empty")
	}
}

func TestChoraLabeledGemini_routesToLoRAAdapterWhenSet(t *testing.T) {
	fake := NewFakeLabeledLLM("gemini-2.5-flash")
	fake.QueueResponse(GenerateResponse{Text: "ok"})

	router := NewInMemoryLoRARouter()
	router.Set("acme", AdapterRef{AdapterPath: "tenant-acme-v3", Version: 3})

	llm, _ := New(Config{Inner: fake, LoRARouter: router, ChoraEnv: "dev"})

	_, err := llm.Generate(context.Background(), GenerateRequest{
		TenantID:  "acme",
		UserGCID:  "u-1",
		CrewName:  "qgen",
		AgentRole: "generation",
		Prompt:    "x",
	})
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}

	call := fake.Calls()[0]
	if call.AdapterPath != "tenant-acme-v3" {
		t.Errorf("want LoRA adapter applied; got %q", call.AdapterPath)
	}
	if call.Labels["chora_lora_adapter"] != "tenant-acme-v3" {
		t.Errorf("want chora_lora_adapter label; got %q", call.Labels["chora_lora_adapter"])
	}
}

func TestChoraLabeledGemini_doesNotSetAdapterLabelWhenBaseModel(t *testing.T) {
	fake := NewFakeLabeledLLM("gemini-2.5-flash")
	fake.QueueResponse(GenerateResponse{Text: "ok"})

	llm, _ := New(Config{Inner: fake, LoRARouter: NewNoopLoRARouter(), ChoraEnv: "dev"})

	_, _ = llm.Generate(context.Background(), GenerateRequest{
		TenantID: "globex", UserGCID: "u", CrewName: "c", AgentRole: "r", Prompt: "x",
	})

	call := fake.Calls()[0]
	if call.AdapterPath != "" {
		t.Errorf("want base model (no adapter); got %q", call.AdapterPath)
	}
	if _, ok := call.Labels["chora_lora_adapter"]; ok {
		t.Errorf("chora_lora_adapter label must be omitted for base model; got %q", call.Labels["chora_lora_adapter"])
	}
}

func TestChoraLabeledGemini_emitsModelInvokedEvent(t *testing.T) {
	fake := NewFakeLabeledLLM("gemini-2.5-flash")
	fake.QueueResponse(GenerateResponse{
		Text:         "answer",
		InputTokens:  42,
		OutputTokens: 17,
	})
	sink := NewRecordingEventSink()

	llm, _ := New(Config{
		Inner:      fake,
		LoRARouter: NewNoopLoRARouter(),
		ChoraEnv:   "dev",
		EventSink:  sink,
	})

	_, err := llm.Generate(context.Background(), GenerateRequest{
		TenantID:   "acme",
		UserGCID:   "u-1",
		CrewName:   "familiar",
		AgentRole:  "companion",
		WorkflowID: "wf-1",
		Prompt:     "hi",
	})
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}

	events := sink.Events()
	if len(events) != 1 {
		t.Fatalf("want 1 emitted event; got %d", len(events))
	}
	ev := events[0]
	if ev.EventType != "model.invoked.v1" {
		t.Errorf("want model.invoked.v1; got %q", ev.EventType)
	}
	if ev.TenantID != "acme" {
		t.Errorf("want tenant acme; got %q", ev.TenantID)
	}
	if ev.UserGCID != "u-1" {
		t.Errorf("want user u-1; got %q", ev.UserGCID)
	}
	if ev.Model != "gemini-2.5-flash" {
		t.Errorf("want model gemini-2.5-flash; got %q", ev.Model)
	}
	if ev.InputTokens != 42 {
		t.Errorf("want input_tokens 42; got %d", ev.InputTokens)
	}
	if ev.OutputTokens != 17 {
		t.Errorf("want output_tokens 17; got %d", ev.OutputTokens)
	}
	if ev.LatencyMS < 0 {
		t.Errorf("latency must be >= 0; got %d", ev.LatencyMS)
	}
}

func TestChoraLabeledGemini_emitsEventEvenOnError(t *testing.T) {
	fake := NewFakeLabeledLLM("gemini-2.5-flash")
	wantErr := errors.New("vertex 503")
	fake.QueueError(wantErr)
	sink := NewRecordingEventSink()

	llm, _ := New(Config{
		Inner:      fake,
		LoRARouter: NewNoopLoRARouter(),
		ChoraEnv:   "dev",
		EventSink:  sink,
	})

	_, err := llm.Generate(context.Background(), GenerateRequest{
		TenantID: "acme", UserGCID: "u", CrewName: "c", AgentRole: "r", Prompt: "x",
	})
	if err == nil {
		t.Fatal("want error from inner LLM; got nil")
	}

	events := sink.Events()
	if len(events) != 1 {
		t.Fatalf("want 1 event emitted even on failure; got %d", len(events))
	}
	if events[0].ErrorCode == "" {
		t.Error("event must capture ErrorCode on failure")
	}
}

func TestChoraLabeledGemini_rejectsMissingTenantID(t *testing.T) {
	llm, _ := New(Config{
		Inner:      NewFakeLabeledLLM("gemini-test"),
		LoRARouter: NewNoopLoRARouter(),
		ChoraEnv:   "dev",
	})

	_, err := llm.Generate(context.Background(), GenerateRequest{
		TenantID: "", UserGCID: "u", CrewName: "c", AgentRole: "r", Prompt: "x",
	})
	if err == nil {
		t.Fatal("want error when TenantID missing; got nil")
	}
	if !strings.Contains(err.Error(), "tenant") {
		t.Errorf("error should mention tenant; got %v", err)
	}
}

func TestChoraLabeledGemini_rejectsMissingUserGCID(t *testing.T) {
	llm, _ := New(Config{
		Inner:      NewFakeLabeledLLM("gemini-test"),
		LoRARouter: NewNoopLoRARouter(),
		ChoraEnv:   "dev",
	})

	_, err := llm.Generate(context.Background(), GenerateRequest{
		TenantID: "acme", UserGCID: "", CrewName: "c", AgentRole: "r", Prompt: "x",
	})
	if err == nil {
		t.Fatal("want error when UserGCID missing; got nil")
	}
}

func TestChoraLabeledGemini_rejectsMissingCrewName(t *testing.T) {
	llm, _ := New(Config{
		Inner:      NewFakeLabeledLLM("gemini-test"),
		LoRARouter: NewNoopLoRARouter(),
		ChoraEnv:   "dev",
	})

	_, err := llm.Generate(context.Background(), GenerateRequest{
		TenantID: "acme", UserGCID: "u", CrewName: "", AgentRole: "r", Prompt: "x",
	})
	if err == nil {
		t.Fatal("want error when CrewName missing; got nil")
	}
}

func TestChoraLabeledGemini_failOpenOnLoRARouterError(t *testing.T) {
	// Per ADR-146 Resilience: if LoRA router fails, fall back to base
	// model. Adapter resolution failure must not block the call.
	fake := NewFakeLabeledLLM("gemini-2.5-flash")
	fake.QueueResponse(GenerateResponse{Text: "ok"})
	router := NewInMemoryLoRARouter()
	router.SetError(errors.New("cloud sql timeout"))

	llm, _ := New(Config{Inner: fake, LoRARouter: router, ChoraEnv: "dev"})

	_, err := llm.Generate(context.Background(), GenerateRequest{
		TenantID: "acme", UserGCID: "u", CrewName: "c", AgentRole: "r", Prompt: "x",
	})
	if err != nil {
		t.Errorf("LoRA router failure must NOT block call (fall back to base model); got %v", err)
	}
	call := fake.Calls()[0]
	if call.AdapterPath != "" {
		t.Errorf("on router error, fall back to base model; got %q", call.AdapterPath)
	}
}

func TestChoraLabeledGemini_preservesCallerLabels(t *testing.T) {
	// Crew-supplied labels (e.g., per-Familiar specialization) must
	// merge with the Chora-injected ones, not be replaced.
	fake := NewFakeLabeledLLM("gemini-2.5-flash")
	fake.QueueResponse(GenerateResponse{Text: "ok"})

	llm, _ := New(Config{Inner: fake, LoRARouter: NewNoopLoRARouter(), ChoraEnv: "dev"})

	_, _ = llm.Generate(context.Background(), GenerateRequest{
		TenantID: "acme", UserGCID: "u", CrewName: "familiar", AgentRole: "companion", Prompt: "x",
		ExtraLabels: map[string]string{
			"familiar_id":             "math-uuid",
			"familiar_specialization": "math",
		},
	})

	call := fake.Calls()[0]
	if call.Labels["familiar_id"] != "math-uuid" {
		t.Errorf("caller label familiar_id must survive merge; got %q", call.Labels["familiar_id"])
	}
	if call.Labels["chora_tenant_id"] != "acme" {
		t.Errorf("chora_tenant_id must still be injected; got %q", call.Labels["chora_tenant_id"])
	}
}

func TestChoraLabeledGemini_rejectsNilInner(t *testing.T) {
	_, err := New(Config{Inner: nil, LoRARouter: NewNoopLoRARouter(), ChoraEnv: "dev"})
	if err == nil {
		t.Fatal("want error for nil Inner; got nil")
	}
}

func TestChoraLabeledGemini_defaultsLoRARouterToNoop(t *testing.T) {
	// LoRARouter is optional — when nil, behaviour is equivalent to NoopLoRARouter.
	fake := NewFakeLabeledLLM("gemini-test")
	fake.QueueResponse(GenerateResponse{Text: "ok"})

	llm, err := New(Config{Inner: fake, LoRARouter: nil, ChoraEnv: "dev"})
	if err != nil {
		t.Fatalf("nil LoRARouter must default to noop; got %v", err)
	}
	_, err = llm.Generate(context.Background(), GenerateRequest{
		TenantID: "acme", UserGCID: "u", CrewName: "c", AgentRole: "r", Prompt: "x",
	})
	if err != nil {
		t.Errorf("unexpected: %v", err)
	}
}

func TestChoraLabeledGemini_modelNameSurfacedFromInner(t *testing.T) {
	fake := NewFakeLabeledLLM("gemini-3-flash-preview")
	llm, _ := New(Config{Inner: fake, LoRARouter: NewNoopLoRARouter(), ChoraEnv: "dev"})

	if llm.Model() != "gemini-3-flash-preview" {
		t.Errorf("Model() should reflect inner LLM; got %q", llm.Model())
	}
}

// ---------- EventSink ----------

func TestRecordingEventSink_recordsAllEvents(t *testing.T) {
	sink := NewRecordingEventSink()

	if err := sink.Emit(context.Background(), ModelInvokedEvent{
		EventType: "model.invoked.v1",
		TenantID:  "acme",
	}); err != nil {
		t.Fatalf("unexpected: %v", err)
	}

	if len(sink.Events()) != 1 {
		t.Fatalf("want 1 event; got %d", len(sink.Events()))
	}
}

func TestNullEventSink_acceptsButDiscardsEvents(t *testing.T) {
	sink := NewNullEventSink()
	err := sink.Emit(context.Background(), ModelInvokedEvent{EventType: "x"})
	if err != nil {
		t.Errorf("null sink must never error; got %v", err)
	}
}
