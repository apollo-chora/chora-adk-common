// Package modelgatewayclient_test — unit tests for the gateway client.
// Uses an in-process bufconn gRPC server that records inbound InvokeRequest
// and returns scripted responses; no live gateway dependency.
package modelgatewayclient

import (
	"context"
	"errors"
	"math"
	"net"
	"testing"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	oteltrace "go.opentelemetry.io/otel/trace"
	"google.golang.org/genai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"

	mgv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/services/model_gateway/v1"

	adkmodel "google.golang.org/adk/model"
)

// fakeServer captures inbound InvokeRequests + emits a scripted response.
type fakeServer struct {
	mgv1.UnimplementedModelGatewayServiceServer

	lastReq *mgv1.InvokeRequest
	lastMD  metadata.MD
	resp    *mgv1.InvokeResponse
	err     error
}

func (f *fakeServer) Invoke(ctx context.Context, in *mgv1.InvokeRequest) (*mgv1.InvokeResponse, error) {
	f.lastReq = in
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		f.lastMD = md
	}
	if f.err != nil {
		return nil, f.err
	}
	if f.resp != nil {
		return f.resp, nil
	}
	return &mgv1.InvokeResponse{
		InvocationId:   "01h0000000000000000000000a",
		Completion:     "B (443) — HTTPS canonical port.",
		Vendor:         "vertex_gemini",
		ModelVersion:   "gemini-2.5-pro",
		LatencyMs:      1234,
		FinishDetail:   "STOP",
		GatewayVersion: "test-0.0.1",
		CompletedAt:    timestamppb.Now(),
	}, nil
}

// startFakeServer spins up a bufconn-backed gRPC server with the given fake
// implementation. Returns the dial option callers can pass via
// Config.dialOptsExtra and a teardown function.
func startFakeServer(t *testing.T, fake *fakeServer) (grpc.DialOption, func()) {
	t.Helper()
	lis := bufconn.Listen(1 << 20)
	srv := grpc.NewServer()
	mgv1.RegisterModelGatewayServiceServer(srv, fake)
	go func() { _ = srv.Serve(lis) }()
	dialer := func(context.Context, string) (net.Conn, error) {
		return lis.Dial()
	}
	opt := grpc.WithContextDialer(dialer)
	return opt, func() {
		srv.GracefulStop()
		_ = lis.Close()
	}
}

func mustNewClient(t *testing.T, fake *fakeServer, override func(*Config)) adkmodel.LLM {
	t.Helper()
	dialerOpt, _ := startFakeServer(t, fake)
	cfg := Config{
		// passthrough:/// scheme tells gRPC to skip DNS resolution; the
		// bufconn dialer captures the connection regardless of the
		// authority part. Required since grpc.NewClient strictly resolves
		// targets via the configured resolver.
		Endpoint:       "passthrough:///bufnet",
		LogicalModelID: "gemini-2.5-pro",
		AgentID:        "qgen_question",
		CrewKind:       "qgen",
		Surface:        "qgen",
		TenantID:       "01k0000000000000000000ten",
		GCID:           "01k0000000000000000000gcid",
		dialOptsExtra: []grpc.DialOption{
			grpc.WithTransportCredentials(insecure.NewCredentials()),
			dialerOpt,
		},
		CallTimeout: 5 * time.Second,
	}
	if override != nil {
		override(&cfg)
	}
	llm, err := New(context.Background(), cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return llm
}

func noopUnary(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
	return invoker(ctx, method, req, reply, cc, opts...)
}

// TestGenerateContent_PropagatesActiveSpanTraceparent guards the agent->gateway
// trace-correlation fix (HANDOFF_OBSERVABILITY_OUTBOX_JAM_2026-05-29 trace wave).
// An ADK agent calling the gateway has NO inbound gRPC metadata; its trace
// context lives in the active OTel span on ctx. The client must propagate THAT
// (via the otelgrpc client handler injecting the W3C propagator into outgoing
// metadata, and onto the proto Traceparent field) so the gateway's Invoke span
// becomes a child of the agent span — one connected trace. Previously the
// client read metadata.FromIncomingContext (always empty here) and propagated
// nothing, leaving the gateway span a standalone root.
func TestGenerateContent_PropagatesActiveSpanTraceparent(t *testing.T) {
	// Global W3C propagator (tracing.Init sets this in production).
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{}, propagation.Baggage{}))

	traceID, _ := oteltrace.TraceIDFromHex("0af7651916cd43dd8448eb211c80319c")
	spanID, _ := oteltrace.SpanIDFromHex("b7ad6b7169203331")
	sc := oteltrace.NewSpanContext(oteltrace.SpanContextConfig{
		TraceID: traceID, SpanID: spanID, TraceFlags: oteltrace.FlagsSampled, Remote: true,
	})
	ctx := oteltrace.ContextWithSpanContext(context.Background(), sc)

	fake := &fakeServer{}
	llm := mustNewClient(t, fake, nil)

	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "what is 2+2?"}}}},
	}
	for _, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			t.Fatalf("GenerateContent yielded error: %v", err)
		}
	}

	// (a) Outgoing gRPC metadata carries a traceparent for the active trace.
	tps := fake.lastMD.Get("traceparent")
	if len(tps) == 0 || tps[0] == "" {
		t.Fatalf("no traceparent propagated in outgoing metadata; md=%v", fake.lastMD)
	}
	if !contains(tps[0], "0af7651916cd43dd8448eb211c80319c") {
		t.Errorf("metadata traceparent %q does not carry the active trace id", tps[0])
	}
	// (b) The proto Traceparent field also carries the active-span trace id
	//     (explicit channel, sourced from the span — NOT inbound metadata).
	if !contains(fake.lastReq.Traceparent, "0af7651916cd43dd8448eb211c80319c") {
		t.Errorf("proto Traceparent %q does not carry the active trace id", fake.lastReq.Traceparent)
	}
}

// ---------------------------------------------------------------------------
// New(...) input validation
// ---------------------------------------------------------------------------

func TestNew_RequiredFields(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		mut  func(*Config)
		want string
	}{
		{"endpoint", func(c *Config) { c.Endpoint = "" }, "Endpoint required"},
		{"logical model", func(c *Config) { c.LogicalModelID = "" }, "LogicalModelID required"},
		{"agent_id", func(c *Config) { c.AgentID = "" }, "AgentID required"},
		{"surface", func(c *Config) { c.Surface = "" }, "Surface required"},
		{"tenant_id", func(c *Config) { c.TenantID = "" }, "TenantID required"},
		{"gcid", func(c *Config) { c.GCID = "" }, "GCID required"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			cfg := Config{
				Endpoint:       "bufnet",
				LogicalModelID: "gemini-2.5-pro",
				AgentID:        "qgen_question",
				Surface:        "qgen",
				TenantID:       "tenant",
				GCID:           "gcid",
				dialOptsExtra:  []grpc.DialOption{grpc.WithTransportCredentials(insecure.NewCredentials())},
			}
			tc.mut(&cfg)
			_, err := New(context.Background(), cfg)
			if err == nil || !contains(err.Error(), tc.want) {
				t.Fatalf("New(%s): got %v, want substring %q", tc.name, err, tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// flattenContents — multi-turn / system / user separation
// ---------------------------------------------------------------------------

func TestFlattenContents(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name     string
		in       []*genai.Content
		wantUser string
		wantSys  string
	}{
		{
			"user only",
			[]*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "Hello"}}}},
			"Hello", "",
		},
		{
			"system + user",
			[]*genai.Content{
				{Role: "system", Parts: []*genai.Part{{Text: "You are an MCQ generator."}}},
				{Role: "user", Parts: []*genai.Part{{Text: "Generate an MCQ."}}},
			},
			"Generate an MCQ.", "You are an MCQ generator.",
		},
		{
			"multi-turn (user → model → user)",
			[]*genai.Content{
				{Role: "user", Parts: []*genai.Part{{Text: "First turn"}}},
				{Role: "model", Parts: []*genai.Part{{Text: "Reply"}}},
				{Role: "user", Parts: []*genai.Part{{Text: "Follow-up"}}},
			},
			"First turn\n\nAssistant: Reply\n\nFollow-up", "",
		},
		{
			"empty role defaults to user",
			[]*genai.Content{{Role: "", Parts: []*genai.Part{{Text: "Bare content"}}}},
			"Bare content", "",
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			gotU, gotS := flattenContents(tc.in)
			if gotU != tc.wantUser {
				t.Errorf("user: got %q, want %q", gotU, tc.wantUser)
			}
			if gotS != tc.wantSys {
				t.Errorf("sys: got %q, want %q", gotS, tc.wantSys)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// GenerateContent — happy path: InvokeRequest fields populated, response
// translated to LLMResponse correctly.
// ---------------------------------------------------------------------------

func TestGenerateContent_HappyPath(t *testing.T) {
	t.Parallel()
	fake := &fakeServer{}
	llm := mustNewClient(t, fake, nil)

	req := &adkmodel.LLMRequest{
		Model: "gemini-2.5-flash",
		Contents: []*genai.Content{
			{Role: "system", Parts: []*genai.Part{{Text: "You are an MCQ generator."}}},
			{Role: "user", Parts: []*genai.Part{{Text: "What is the HTTPS port?"}}},
		},
		Config: &genai.GenerateContentConfig{
			Temperature:     ptrFloat32(0.3),
			MaxOutputTokens: 256,
		},
	}

	var responses []*adkmodel.LLMResponse
	for resp, err := range llm.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent yielded error: %v", err)
		}
		responses = append(responses, resp)
	}
	if len(responses) != 1 {
		t.Fatalf("expected 1 response, got %d", len(responses))
	}

	// Verify the InvokeRequest the server received carries every required field.
	got := fake.lastReq
	if got == nil {
		t.Fatal("server did not receive an InvokeRequest")
	}
	if got.TenantId != "01k0000000000000000000ten" {
		t.Errorf("tenant_id: got %q", got.TenantId)
	}
	if got.Gcid != "01k0000000000000000000gcid" {
		t.Errorf("gcid: got %q", got.Gcid)
	}
	if got.AgentId != "qgen_question" {
		t.Errorf("agent_id: got %q", got.AgentId)
	}
	if got.LogicalModelId != "gemini-2.5-flash" { // per-call override wins
		t.Errorf("logical_model_id: got %q, want gemini-2.5-flash (per-call override)", got.LogicalModelId)
	}
	if got.Prompt != "What is the HTTPS port?" {
		t.Errorf("prompt: got %q", got.Prompt)
	}
	if got.SystemPrompt != "You are an MCQ generator." {
		t.Errorf("system_prompt: got %q", got.SystemPrompt)
	}
	if got.GenerationConfig == nil {
		t.Error("generation_config should be set")
	} else {
		m := got.GenerationConfig.AsMap()
		// float32 → float64 round-trip via structpb (JSON-Number) preserves
		// the IEEE-754 representation of float32(0.3), which is not exactly
		// 0.3. Allow a small tolerance.
		if got := m["temperature"].(float64); math.Abs(got-0.3) > 1e-5 {
			t.Errorf("temperature: got %v, want ~0.3", got)
		}
		if m["max_output_tokens"].(float64) != 256 {
			t.Errorf("max_output_tokens: got %v", m["max_output_tokens"])
		}
	}

	// Verify LLMResponse translation.
	resp := responses[0]
	if resp.Content == nil || len(resp.Content.Parts) != 1 {
		t.Fatalf("response Content not populated: %+v", resp.Content)
	}
	if resp.Content.Role != "model" {
		t.Errorf("response role: got %q", resp.Content.Role)
	}
	if resp.Content.Parts[0].Text != "B (443) — HTTPS canonical port." {
		t.Errorf("response text: got %q", resp.Content.Parts[0].Text)
	}
	if resp.ModelVersion != "gemini-2.5-pro" {
		t.Errorf("ModelVersion: got %q", resp.ModelVersion)
	}
	if resp.FinishReason != genai.FinishReasonStop {
		t.Errorf("FinishReason: got %v, want Stop", resp.FinishReason)
	}
	if !resp.TurnComplete {
		t.Error("TurnComplete should be true for unary")
	}
	if resp.CustomMetadata["chora.gateway.vendor"] != "vertex_gemini" {
		t.Errorf("vendor metadata: got %v", resp.CustomMetadata["chora.gateway.vendor"])
	}
}

// ---------------------------------------------------------------------------
// GenerateContent — per-turn instruction from Config.SystemInstruction.
//
// Regression for the 2026-05-25 smoke that reached APPROVED status but
// produced a generic LLM reply "I'm not sure what the situation is
// about. I'm waiting for you to tell me." The qgen crew sets the
// per-turn instruction via WithInstruction(...) / InstructionProvider
// callback, which ADK routes through req.Config.SystemInstruction (NOT
// req.Contents with role="system"). Earlier modelgatewayclient ignored
// the field, so the model received ADK's "Handle the requests as
// specified in the System Instruction." placeholder user content with
// NO system instruction. Propagating the field is mandatory.
// ---------------------------------------------------------------------------

func TestGenerateContent_SystemInstructionFromConfig(t *testing.T) {
	t.Parallel()
	fake := &fakeServer{}
	llm := mustNewClient(t, fake, nil)

	instructionText := "You are an MCQ generator. Produce JSON {stem, options[4], correct}."
	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "What is the OSI layer of TCP?"}}},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Role:  "system",
				Parts: []*genai.Part{{Text: instructionText}},
			},
		},
	}

	for _, err := range llm.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent yielded error: %v", err)
		}
	}

	got := fake.lastReq
	if got == nil {
		t.Fatal("server did not receive an InvokeRequest")
	}
	if got.SystemPrompt != instructionText {
		t.Errorf("system_prompt: got %q, want %q (Config.SystemInstruction must propagate)", got.SystemPrompt, instructionText)
	}
}

// Config.SystemInstruction takes precedence over Contents system role
// when both are present; the Contents system part is appended after so
// no information is lost.
func TestGenerateContent_SystemInstruction_MergesWithContentsSystem(t *testing.T) {
	t.Parallel()
	fake := &fakeServer{}
	llm := mustNewClient(t, fake, nil)

	configInstr := "You generate MCQs."
	contentsInstr := "Use IEEE citation format."
	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{
			{Role: "system", Parts: []*genai.Part{{Text: contentsInstr}}},
			{Role: "user", Parts: []*genai.Part{{Text: "TCP OSI layer?"}}},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Role:  "system",
				Parts: []*genai.Part{{Text: configInstr}},
			},
		},
	}

	for _, err := range llm.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent yielded error: %v", err)
		}
	}

	want := configInstr + "\n\n" + contentsInstr
	if got := fake.lastReq.SystemPrompt; got != want {
		t.Errorf("system_prompt: got %q, want %q", got, want)
	}
}

// ---------------------------------------------------------------------------
// GenerateContent — CustomMetadata round-trips through structpb.
//
// Regression for the 2026-05-25 production bug: ADK stamps the
// LLMResponse onto the session event via structpb.NewValue, which rejects
// concrete-slice types ("proto: invalid type: []string"). The whole
// metadata map MUST serialize cleanly even when the gateway returned a
// multi-element FallbackChain.
// ---------------------------------------------------------------------------

func TestGenerateContent_CustomMetadata_StructpbSerialisable(t *testing.T) {
	t.Parallel()
	fake := &fakeServer{resp: &mgv1.InvokeResponse{
		InvocationId:   "01h0000000000000000000000a",
		Completion:     "ok",
		Vendor:         "vertex_ai_gemini",
		ModelVersion:   "gemini-2.5-flash",
		FallbackChain:  []string{"vertex_ai_gemini:gemini-2.5-pro", "vertex_ai_gemini:gemini-2.5-flash"},
		LatencyMs:      42,
		FinishDetail:   "STOP",
		GatewayVersion: "test-0.0.1",
		CompletedAt:    timestamppb.Now(),
	}}
	llm := mustNewClient(t, fake, nil)

	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
	}

	var resp *adkmodel.LLMResponse
	for r, err := range llm.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent yielded error: %v", err)
		}
		resp = r
	}
	if resp == nil {
		t.Fatal("no response yielded")
	}

	// The exact ADK call that previously failed in prod: stamp the entire
	// metadata map through structpb.NewValue.
	if _, err := structpb.NewValue(resp.CustomMetadata); err != nil {
		t.Fatalf("CustomMetadata not structpb-serializable: %v\nmap = %#v", err, resp.CustomMetadata)
	}

	// fallback_chain MUST be []any with both vendor:model entries preserved
	// in order so downstream cost-attribution dashboards keep the dispatch
	// trail.
	got, ok := resp.CustomMetadata["chora.gateway.fallback_chain"].([]any)
	if !ok {
		t.Fatalf("fallback_chain wrong type: got %T, want []any", resp.CustomMetadata["chora.gateway.fallback_chain"])
	}
	if len(got) != 2 || got[0] != "vertex_ai_gemini:gemini-2.5-pro" || got[1] != "vertex_ai_gemini:gemini-2.5-flash" {
		t.Errorf("fallback_chain: got %v, want [pro, flash]", got)
	}
}

// Empty fallback chain (happy-path: single-try, no fallback) MUST still
// produce a structpb-friendly value — never nil, never []string.
func TestGenerateContent_CustomMetadata_EmptyFallbackChain(t *testing.T) {
	t.Parallel()
	fake := &fakeServer{} // default resp has nil FallbackChain
	llm := mustNewClient(t, fake, nil)

	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
	}

	var resp *adkmodel.LLMResponse
	for r, err := range llm.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent yielded error: %v", err)
		}
		resp = r
	}

	if _, err := structpb.NewValue(resp.CustomMetadata); err != nil {
		t.Fatalf("CustomMetadata not structpb-serializable on empty fallback chain: %v", err)
	}
	got, ok := resp.CustomMetadata["chora.gateway.fallback_chain"].([]any)
	if !ok {
		t.Fatalf("fallback_chain wrong type: got %T, want []any", resp.CustomMetadata["chora.gateway.fallback_chain"])
	}
	if len(got) != 0 {
		t.Errorf("fallback_chain on happy-path should be empty []any, got %v", got)
	}
}

// ---------------------------------------------------------------------------
// GenerateContent — no user content → error before dispatch.
// ---------------------------------------------------------------------------

func TestGenerateContent_EmptyContents(t *testing.T) {
	t.Parallel()
	fake := &fakeServer{}
	llm := mustNewClient(t, fake, nil)

	req := &adkmodel.LLMRequest{Contents: nil}
	var hadErr bool
	for resp, err := range llm.GenerateContent(context.Background(), req, false) {
		if err != nil {
			hadErr = true
			if !contains(err.Error(), "no user content") {
				t.Errorf("error message: %v", err)
			}
		}
		_ = resp
	}
	if !hadErr {
		t.Error("expected error for empty Contents")
	}
	if fake.lastReq != nil {
		t.Error("server should not be called when Contents are empty")
	}
}

// ---------------------------------------------------------------------------
// GenerateContent — server-side error propagated to caller.
// ---------------------------------------------------------------------------

func TestGenerateContent_ServerError(t *testing.T) {
	t.Parallel()
	fake := &fakeServer{err: errors.New("simulated vendor failure")}
	llm := mustNewClient(t, fake, nil)

	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
	}
	var sawErr error
	for _, err := range llm.GenerateContent(context.Background(), req, false) {
		if err != nil {
			sawErr = err
		}
	}
	if sawErr == nil {
		t.Fatal("expected error from server")
	}
	if !contains(sawErr.Error(), "simulated vendor failure") {
		t.Errorf("error message lost vendor detail: %v", sawErr)
	}
}

// ---------------------------------------------------------------------------
// FinishDetail mapping
// ---------------------------------------------------------------------------

func TestMapFinishDetail(t *testing.T) {
	t.Parallel()
	cases := []struct {
		in   string
		want genai.FinishReason
	}{
		{"STOP", genai.FinishReasonStop},
		{"stop", genai.FinishReasonStop},
		{"", genai.FinishReasonStop},
		{"MAX_TOKENS", genai.FinishReasonMaxTokens},
		{"SAFETY", genai.FinishReasonSafety},
		{"RECITATION", genai.FinishReasonRecitation},
		{"weird-future-value", genai.FinishReasonUnspecified},
	}
	for _, tc := range cases {
		if got := mapFinishDetail(tc.in); got != tc.want {
			t.Errorf("mapFinishDetail(%q): got %v, want %v", tc.in, got, tc.want)
		}
	}
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func ptrFloat32(v float32) *float32 { return &v }

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || indexOf(s, sub) >= 0)
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

// ---------------------------------------------------------------------------
// ADR-254 D7: surface + dispatch_idempotency_key on the wire.
// ---------------------------------------------------------------------------

func TestGenerateContent_StampsSurfaceOnEveryInvoke(t *testing.T) {
	t.Parallel()
	fake := &fakeServer{}
	llm := mustNewClient(t, fake, nil)

	// Turn-initiating request AND a tool-result continuation: the gateway
	// refuses an ABSENT surface before any debit, so both shapes must carry it.
	turn := &adkmodel.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "grade this"}}},
	}}
	continuation := &adkmodel.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "grade this"}}},
		{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "lookup", Args: map[string]any{}}}}},
		{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{Name: "lookup", Response: map[string]any{"ok": true}}}}},
	}}
	for name, r := range map[string]*adkmodel.LLMRequest{"turn": turn, "continuation": continuation} {
		for _, err := range llm.GenerateContent(context.Background(), r, false) {
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
		}
		if fake.lastReq == nil || fake.lastReq.Surface != "qgen" {
			t.Errorf("%s: InvokeRequest.surface = %q, want %q (crew id)", name, fake.lastReq.GetSurface(), "qgen")
		}
	}
}

func TestGenerateContent_ForwardsTheDispatchKeyFromTheLabel(t *testing.T) {
	t.Parallel()
	fake := &fakeServer{}
	llm := mustNewClient(t, fake, nil)

	withKey := &adkmodel.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "grade"}}}},
		Config:   &genai.GenerateContentConfig{Labels: map[string]string{LabelDispatchIdempotencyKey: "agent_dispatch.oe_evaluate.sub-1:q-1:1"}},
	}
	for _, err := range llm.GenerateContent(context.Background(), withKey, false) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if got := fake.lastReq.GetDispatchIdempotencyKey(); got != "agent_dispatch.oe_evaluate.sub-1:q-1:1" {
		t.Errorf("dispatch_idempotency_key = %q, want the label value", got)
	}

	// A non-dispatched call (no label) stamps nothing: the gateway treats an
	// empty key as "not a dispatch" and debits normally.
	without := &adkmodel.LLMRequest{Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "grade"}}}}}
	for _, err := range llm.GenerateContent(context.Background(), without, false) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if got := fake.lastReq.GetDispatchIdempotencyKey(); got != "" {
		t.Errorf("dispatch_idempotency_key = %q on a non-dispatched call, want empty", got)
	}
}
