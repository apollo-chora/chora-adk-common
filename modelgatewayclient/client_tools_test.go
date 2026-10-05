// client_tools_test.go — ADR-177 tool-aware bridge unit tests.
//
// Covers the pure helpers that carry the structured genai conversation + tool
// declarations through Invoke and parse model-emitted tool calls back: contents
// marshaling (with the single-turn flat-path opt-out), tool marshaling, per-turn
// metering turn-detection, and response-part assembly.
package modelgatewayclient

import (
	"encoding/json"
	"strings"
	"testing"

	"google.golang.org/genai"
)

func TestMarshalContents_SingleTurnTextStaysFlat(t *testing.T) {
	t.Parallel()
	// One user text content, no tools → flat path (contents_json empty),
	// byte-identical to the proven qgen/image-gen callers.
	contents := []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}}
	got, err := marshalContents(contents)
	if err != nil {
		t.Fatalf("marshalContents: %v", err)
	}
	if got != "" {
		t.Errorf("single-turn text should stay flat (empty contents_json); got %q", got)
	}
}

func TestMarshalContents_MultiTurnSerialised(t *testing.T) {
	t.Parallel()
	contents := []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "q1"}}},
		{Role: "model", Parts: []*genai.Part{{Text: "a1"}}},
	}
	got, err := marshalContents(contents)
	if err != nil {
		t.Fatalf("marshalContents: %v", err)
	}
	if got == "" {
		t.Fatal("multi-turn must serialise contents_json")
	}
	// The genai JSON shape must round-trip into the gateway's gemini wire shape
	// (role/parts/text). Verify via a structural decode.
	var decoded []struct {
		Role  string `json:"role"`
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	}
	if err := json.Unmarshal([]byte(got), &decoded); err != nil {
		t.Fatalf("contents_json not gemini-shaped: %v", err)
	}
	if len(decoded) != 2 || decoded[0].Role != "user" || decoded[0].Parts[0].Text != "q1" {
		t.Errorf("unexpected decoded contents: %+v", decoded)
	}
	if decoded[1].Role != "model" || decoded[1].Parts[0].Text != "a1" {
		t.Errorf("model turn not preserved: %+v", decoded)
	}
}

func TestMarshalContents_FunctionPartForcesSerialisation(t *testing.T) {
	t.Parallel()
	// A single content that carries a non-text (functionResponse) part must be
	// serialised even though there is only one entry — the flat prompt drops it.
	contents := []*genai.Content{{
		Role:  "user",
		Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{Name: "search_atoms", Response: map[string]any{"hits": 3}}}},
	}}
	got, err := marshalContents(contents)
	if err != nil {
		t.Fatalf("marshalContents: %v", err)
	}
	if got == "" {
		t.Fatal("function-response part must force contents_json serialisation")
	}
}

func TestMarshalContents_FileDataPartForcesSerialisation(t *testing.T) {
	t.Parallel()
	// EPIC-1a batch grounding: a single user content carrying a fileData (gs://)
	// part must serialise to contents_json — the flat prompt would drop the blob
	// reference, so the model would never see the uploaded source material.
	contents := []*genai.Content{{
		Role: "user",
		Parts: []*genai.Part{
			{Text: "Generate MCQs from the attached source."},
			{FileData: &genai.FileData{
				FileURI:  "gs://chora-batch-uploads/t/job/material.pdf",
				MIMEType: "application/pdf",
			}},
		},
	}}
	got, err := marshalContents(contents)
	if err != nil {
		t.Fatalf("marshalContents: %v", err)
	}
	if got == "" {
		t.Fatal("fileData part must force contents_json serialisation")
	}
	if !strings.Contains(got, "gs://chora-batch-uploads/t/job/material.pdf") {
		t.Errorf("contents_json missing the gs:// fileUri: %s", got)
	}
}

func TestMarshalTools_Roundtrip(t *testing.T) {
	t.Parallel()
	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{{
			FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "search_atoms", Description: "search the atom corpus"}},
		}},
	}
	got, err := marshalTools(cfg)
	if err != nil {
		t.Fatalf("marshalTools: %v", err)
	}
	if got == "" {
		t.Fatal("tools must serialise to tools_json")
	}
	var decoded []struct {
		FunctionDeclarations []struct {
			Name string `json:"name"`
		} `json:"functionDeclarations"`
	}
	if err := json.Unmarshal([]byte(got), &decoded); err != nil {
		t.Fatalf("tools_json not vertex-shaped: %v", err)
	}
	if len(decoded) != 1 || decoded[0].FunctionDeclarations[0].Name != "search_atoms" {
		t.Errorf("unexpected decoded tools: %+v", decoded)
	}
}

func TestMarshalTools_NilConfigEmpty(t *testing.T) {
	t.Parallel()
	got, err := marshalTools(nil)
	if err != nil {
		t.Fatalf("marshalTools(nil): %v", err)
	}
	if got != "" {
		t.Errorf("nil config → empty tools_json; got %q", got)
	}
}

func TestIsTurnInitiating(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name     string
		contents []*genai.Content
		want     bool
	}{
		{"empty conversation", nil, true},
		{"user text turn", []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}}, true},
		{"role defaults to user", []*genai.Content{{Parts: []*genai.Part{{Text: "hi"}}}}, true},
		{
			"tool-result continuation",
			[]*genai.Content{
				{Role: "user", Parts: []*genai.Part{{Text: "find atoms"}}},
				{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "search_atoms"}}}},
				{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{Name: "search_atoms"}}}},
			},
			false,
		},
		{
			"trailing model turn",
			[]*genai.Content{{Role: "model", Parts: []*genai.Part{{Text: "a"}}}},
			false,
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := isTurnInitiating(tc.contents); got != tc.want {
				t.Errorf("isTurnInitiating(%s) = %v; want %v", tc.name, got, tc.want)
			}
		})
	}
}

func TestBuildResponseParts_TextOnly(t *testing.T) {
	t.Parallel()
	parts, err := buildResponseParts("hello", "")
	if err != nil {
		t.Fatalf("buildResponseParts: %v", err)
	}
	if len(parts) != 1 || parts[0].Text != "hello" {
		t.Errorf("text-only: got %+v", parts)
	}
}

func TestBuildResponseParts_EmptyCompletionPreserved(t *testing.T) {
	t.Parallel()
	// Pre-ADR-177 shape: a single (possibly empty) text part.
	parts, err := buildResponseParts("", "")
	if err != nil {
		t.Fatalf("buildResponseParts: %v", err)
	}
	if len(parts) != 1 || parts[0].Text != "" {
		t.Errorf("empty completion should yield single empty text part; got %+v", parts)
	}
}

func TestBuildResponseParts_ToolCalls(t *testing.T) {
	t.Parallel()
	// Gateway surfaces the genai [{name,args}] shape (from []geminiFunctionCall).
	tc := `[{"name":"search_atoms","args":{"q":"newton"}},{"name":"cite_atom","args":{"id":"a1"}}]`
	parts, err := buildResponseParts("", tc)
	if err != nil {
		t.Fatalf("buildResponseParts: %v", err)
	}
	if len(parts) != 2 {
		t.Fatalf("want 2 function-call parts; got %d", len(parts))
	}
	if parts[0].FunctionCall == nil || parts[0].FunctionCall.Name != "search_atoms" {
		t.Errorf("part0 not search_atoms call: %+v", parts[0])
	}
	if parts[0].FunctionCall.Args["q"] != "newton" {
		t.Errorf("args not parsed: %+v", parts[0].FunctionCall.Args)
	}
	if parts[1].FunctionCall == nil || parts[1].FunctionCall.Name != "cite_atom" {
		t.Errorf("part1 not cite_atom call: %+v", parts[1])
	}
}

func TestBuildResponseParts_ToolCallsWithText(t *testing.T) {
	t.Parallel()
	parts, err := buildResponseParts("thinking...", `[{"name":"search_atoms","args":{}}]`)
	if err != nil {
		t.Fatalf("buildResponseParts: %v", err)
	}
	if len(parts) != 2 {
		t.Fatalf("want function-call + text parts; got %d", len(parts))
	}
	if parts[0].FunctionCall == nil {
		t.Errorf("first part should be the function call")
	}
	if parts[1].Text != "thinking..." {
		t.Errorf("trailing text not preserved: %+v", parts[1])
	}
}

func TestBuildResponseParts_MalformedToolCallsErrors(t *testing.T) {
	t.Parallel()
	if _, err := buildResponseParts("", "{not-an-array"); err == nil {
		t.Error("malformed tool_calls_json must error (not silently drop the turn)")
	}
}
