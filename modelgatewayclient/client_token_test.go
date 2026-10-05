// Package modelgatewayclient_test — C3 per-hop token surfacing tests.
//
// C3: the gateway's InvokeResponse.usage (TokenUsage{input_tokens,
// output_tokens, cost_micros}) must reach the orchestrator's pipeline_trace.
// The Python executor's _map_response reads `tokens_consumed_total` /
// `input_tokens` / `output_tokens` off the TERMINAL CANDIDATE JSON (the LLM
// completion text). So the gateway client must (a) surface usage into the
// in-process LLMResponse.UsageMetadata + CustomMetadata AND (b) merge the
// three token fields into the completion's top-level JSON object so the
// executor can read them off the terminal text.
//
// Before C3 the client dropped resp.GetUsage() entirely — the executor saw
// tokens_consumed_total=0 and the W6 token-compounding eval could not compute.
package modelgatewayclient

import (
	"context"
	"encoding/json"
	"testing"

	"google.golang.org/genai"
	"google.golang.org/protobuf/types/known/timestamppb"

	mgv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/services/model_gateway/v1"

	adkmodel "google.golang.org/adk/model"
)

// respWithUsage returns a scripted InvokeResponse carrying a candidate JSON
// completion + a populated TokenUsage block.
func respWithUsage(completion string, in, out, cached int64) *mgv1.InvokeResponse {
	return &mgv1.InvokeResponse{
		InvocationId: "01h0000000000000000000000a",
		Completion:   completion,
		Usage: &mgv1.TokenUsage{
			InputTokens:  in,
			OutputTokens: out,
			CachedTokens: cached,
			CostMicros:   4242,
		},
		Vendor:         "vertex_ai_gemini",
		ModelVersion:   "gemini-3.1-pro-preview",
		LatencyMs:      77,
		FinishDetail:   "STOP",
		GatewayVersion: "test-0.0.1",
		CompletedAt:    timestamppb.Now(),
	}
}

func runOnce(t *testing.T, fake *fakeServer) *adkmodel.LLMResponse {
	t.Helper()
	llm := mustNewClient(t, fake, nil)
	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "generate"}}}},
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
	return resp
}

// C3 (a) — usage surfaces into LLMResponse.UsageMetadata (genai shape).
func TestGenerateContent_SurfacesUsageMetadata(t *testing.T) {
	t.Parallel()
	candidate := `{"candidate":{"stem":"Q?","options":[],"intent":"new_question","question_type":"mcq"}}`
	fake := &fakeServer{resp: respWithUsage(candidate, 123, 45, 7)}
	resp := runOnce(t, fake)

	if resp.UsageMetadata == nil {
		t.Fatal("UsageMetadata nil — gateway usage was dropped")
	}
	if got := resp.UsageMetadata.PromptTokenCount; got != 123 {
		t.Errorf("PromptTokenCount: got %d, want 123", got)
	}
	if got := resp.UsageMetadata.CandidatesTokenCount; got != 45 {
		t.Errorf("CandidatesTokenCount: got %d, want 45", got)
	}
	if got := resp.UsageMetadata.TotalTokenCount; got != 168 {
		t.Errorf("TotalTokenCount: got %d, want 168 (123+45)", got)
	}
	if got := resp.UsageMetadata.CachedContentTokenCount; got != 7 {
		t.Errorf("CachedContentTokenCount: got %d, want 7", got)
	}
}

// C3 (a) — usage surfaces into CustomMetadata (structpb-serialisable) for
// ai-cost-tracking + OTLP visibility. The whole map must serialise (the
// 2026-05-25 prod bug).
func TestGenerateContent_SurfacesUsageCustomMetadata(t *testing.T) {
	t.Parallel()
	candidate := `{"candidate":{"stem":"Q?","question_type":"mcq"}}`
	fake := &fakeServer{resp: respWithUsage(candidate, 100, 50, 0)}
	resp := runOnce(t, fake)

	if got := resp.CustomMetadata["chora.gateway.input_tokens"]; got != int64(100) {
		t.Errorf("input_tokens metadata: got %v (%T), want int64(100)", got, got)
	}
	if got := resp.CustomMetadata["chora.gateway.output_tokens"]; got != int64(50) {
		t.Errorf("output_tokens metadata: got %v (%T), want int64(50)", got, got)
	}
	if got := resp.CustomMetadata["chora.gateway.tokens_consumed_total"]; got != int64(150) {
		t.Errorf("tokens_consumed_total metadata: got %v (%T), want int64(150)", got, got)
	}
	if got := resp.CustomMetadata["chora.gateway.cost_micros"]; got != int64(4242) {
		t.Errorf("cost_micros metadata: got %v (%T), want int64(4242)", got, got)
	}
}

// C3 (b) — the three token fields are merged into the completion's top-level
// JSON object so the Python executor's _map_response reads them off the
// terminal candidate text. tokens_consumed_total = input + output.
func TestGenerateContent_MergesTokensIntoCompletionJSON(t *testing.T) {
	t.Parallel()
	candidate := `{"candidate":{"stem":"What is 2+2?","question_type":"mcq"}}`
	fake := &fakeServer{resp: respWithUsage(candidate, 200, 60, 0)}
	resp := runOnce(t, fake)

	if resp.Content == nil || len(resp.Content.Parts) != 1 {
		t.Fatalf("response Content not populated: %+v", resp.Content)
	}
	text := resp.Content.Parts[0].Text

	var merged map[string]any
	if err := json.Unmarshal([]byte(text), &merged); err != nil {
		t.Fatalf("merged completion is not valid JSON: %v\ntext=%s", err, text)
	}
	if got := asInt(merged["input_tokens"]); got != 200 {
		t.Errorf("merged input_tokens: got %v, want 200", merged["input_tokens"])
	}
	if got := asInt(merged["output_tokens"]); got != 60 {
		t.Errorf("merged output_tokens: got %v, want 60", merged["output_tokens"])
	}
	if got := asInt(merged["tokens_consumed_total"]); got != 260 {
		t.Errorf("merged tokens_consumed_total: got %v, want 260 (200+60)", merged["tokens_consumed_total"])
	}
	// The original candidate payload MUST be preserved verbatim alongside the
	// token fields — token injection is additive, never destructive.
	cand, ok := merged["candidate"].(map[string]any)
	if !ok {
		t.Fatalf("merged JSON lost the candidate object: %#v", merged)
	}
	if cand["stem"] != "What is 2+2?" {
		t.Errorf("candidate.stem altered: got %v", cand["stem"])
	}
	// CRITICAL unwrap-survival contract: the generation agent wraps its
	// candidate under "candidate"; the Python executor's _map_response unwraps
	// (raw = raw["candidate"]) BEFORE reading tokens_consumed_total. So the
	// token fields MUST also be present INSIDE the nested candidate, else the
	// unwrap discards them and the executor reads 0 again (the exact pre-C3
	// bug). Assert the nested copy is there.
	if asInt(cand["tokens_consumed_total"]) != 260 {
		t.Errorf("nested candidate.tokens_consumed_total: got %v, want 260 (must survive Python _map_response unwrap)", cand["tokens_consumed_total"])
	}
	if asInt(cand["input_tokens"]) != 200 || asInt(cand["output_tokens"]) != 60 {
		t.Errorf("nested candidate per-hop tokens wrong: in=%v out=%v", cand["input_tokens"], cand["output_tokens"])
	}
}

// C3 (b) — flat candidate (no "candidate" wrapper, e.g. the qgen_critic
// {"accepted":...} output) gets the token fields at the top level only — that
// is where the Python critic node reads them (no unwrap happens for the
// critic). Guards that the critic hop surfaces its own usage.
func TestGenerateContent_MergesTokensIntoFlatCriticJSON(t *testing.T) {
	t.Parallel()
	critique := `{"accepted":true,"critique_notes":"","suggested_revisions":[]}`
	fake := &fakeServer{resp: respWithUsage(critique, 90, 30, 0)}
	resp := runOnce(t, fake)

	var merged map[string]any
	if err := json.Unmarshal([]byte(resp.Content.Parts[0].Text), &merged); err != nil {
		t.Fatalf("merged critic JSON invalid: %v", err)
	}
	if asInt(merged["tokens_consumed_total"]) != 120 {
		t.Errorf("critic tokens_consumed_total: got %v, want 120", merged["tokens_consumed_total"])
	}
	if asInt(merged["input_tokens"]) != 90 || asInt(merged["output_tokens"]) != 30 {
		t.Errorf("critic per-hop tokens wrong: in=%v out=%v", merged["input_tokens"], merged["output_tokens"])
	}
	// Critic decision fields preserved verbatim.
	if merged["accepted"] != true {
		t.Errorf("critic accepted field altered: got %v", merged["accepted"])
	}
}

// C3 (b) — markdown-fenced completion (a Gemini emission quirk) still gets the
// token fields merged: the fence is peeled, the inner object augmented, and the
// fence restored so the Python _strip_markdown_fences + _parse_terminal_json
// path still recovers the object.
func TestGenerateContent_MergesTokensIntoFencedCompletionJSON(t *testing.T) {
	t.Parallel()
	fenced := "```json\n{\"candidate\":{\"stem\":\"Q?\",\"question_type\":\"oe\"}}\n```"
	fake := &fakeServer{resp: respWithUsage(fenced, 80, 20, 0)}
	resp := runOnce(t, fake)

	text := resp.Content.Parts[0].Text
	peeled := stripFenceForTest(text)
	var merged map[string]any
	if err := json.Unmarshal([]byte(peeled), &merged); err != nil {
		t.Fatalf("fenced merged completion inner JSON invalid: %v\ntext=%s", err, text)
	}
	if asInt(merged["tokens_consumed_total"]) != 100 {
		t.Errorf("fenced merge tokens_consumed_total: got %v, want 100", merged["tokens_consumed_total"])
	}
	if asInt(merged["input_tokens"]) != 80 || asInt(merged["output_tokens"]) != 20 {
		t.Errorf("fenced merge per-hop tokens wrong: in=%v out=%v", merged["input_tokens"], merged["output_tokens"])
	}
}

// C3 (b) — fail-soft: a non-JSON completion (e.g. a block message or plain
// prose) MUST pass through verbatim. Token injection must never corrupt a
// non-JSON candidate; the Python side simply reads tokens_consumed_total=0 on
// that non-happy path. UsageMetadata is still surfaced.
func TestGenerateContent_NonJSONCompletionPassesThroughVerbatim(t *testing.T) {
	t.Parallel()
	prose := "I cannot generate that question."
	fake := &fakeServer{resp: respWithUsage(prose, 10, 0, 0)}
	resp := runOnce(t, fake)

	if got := resp.Content.Parts[0].Text; got != prose {
		t.Errorf("non-JSON completion was mutated: got %q, want %q", got, prose)
	}
	// Usage still surfaced in-process even though the completion wasn't JSON.
	if resp.UsageMetadata == nil || resp.UsageMetadata.PromptTokenCount != 10 {
		t.Errorf("UsageMetadata not surfaced on non-JSON completion: %+v", resp.UsageMetadata)
	}
}

// C3 — zero-usage (gateway returned no usage block, e.g. nil Usage) must not
// inject token fields and must not crash. Today's byte-compatible behaviour:
// the completion JSON is returned unchanged (no zero-valued token keys forced
// in), so a real candidate without usage data stays identical to pre-C3.
func TestGenerateContent_NilUsageLeavesCompletionUnchanged(t *testing.T) {
	t.Parallel()
	candidate := `{"candidate":{"stem":"Q?","question_type":"mcq"}}`
	fake := &fakeServer{resp: &mgv1.InvokeResponse{
		InvocationId: "01h0000000000000000000000a",
		Completion:   candidate,
		// Usage deliberately nil.
		Vendor:         "vertex_ai_gemini",
		ModelVersion:   "gemini-3.1-pro-preview",
		FinishDetail:   "STOP",
		GatewayVersion: "test-0.0.1",
		CompletedAt:    timestamppb.Now(),
	}}
	resp := runOnce(t, fake)

	if got := resp.Content.Parts[0].Text; got != candidate {
		t.Errorf("nil-usage completion was mutated: got %q, want %q", got, candidate)
	}
	if resp.UsageMetadata != nil {
		t.Errorf("UsageMetadata should be nil when gateway returns no usage; got %+v", resp.UsageMetadata)
	}
}

// asInt coerces a JSON-decoded numeric (float64) to int for assertions.
func asInt(v any) int {
	switch x := v.(type) {
	case float64:
		return int(x)
	case int:
		return x
	case int64:
		return int(x)
	}
	return -1
}

// stripFenceForTest mirrors the Python _strip_markdown_fences enough for the
// fenced-merge assertion (peels a leading ```json / ``` fence + trailing ```).
func stripFenceForTest(s string) string {
	return stripMarkdownFence(s)
}
