package groundingplugin

import (
	"errors"
	"strings"
	"testing"

	adkmodel "google.golang.org/adk/model"
	"google.golang.org/genai"
)

// fakeState is a minimal session-state stub. A missing key returns (nil, nil)
// → readState yields "" (the live miss behaviour).
type fakeState map[string]any

func (f fakeState) Get(k string) (any, error) { return f[k], nil }

// errState fails every Get (transient state-store error).
type errState struct{}

func (errState) Get(string) (any, error) { return nil, errors.New("state unavailable") }

func userReq(text string) *adkmodel.LLMRequest {
	return &adkmodel.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: text}}}},
	}
}

func TestApplyGrounding_InjectsTextPart(t *testing.T) {
	req := userReq("Generate MCQs.")
	state := fakeState{"source_material_text": `<SOURCE file="paper.pdf" page="1">Photosynthesis.</SOURCE>`}

	if err := applyGroundingFromState(state, req); err != nil {
		t.Fatalf("applyGroundingFromState: %v", err)
	}

	parts := req.Contents[0].Parts
	if len(parts) != 2 {
		t.Fatalf("parts = %d; want 2 (instruction + grounding text)", len(parts))
	}
	if parts[1].FileData != nil {
		t.Fatalf("FileData injected; the gateway cannot carry it")
	}
	if !strings.Contains(parts[1].Text, "Photosynthesis.") {
		t.Errorf("grounding text = %q; want the material preserved", parts[1].Text)
	}
	if !strings.HasPrefix(parts[1].Text, "\n\n") {
		t.Error("grounding text should be separated from the author's instruction")
	}
}

// The whole point of this package's change: no part may ever be a blob
// reference, even when the state still declares files.
func TestApplyGrounding_NeverEmitsFileData(t *testing.T) {
	req := userReq("Generate MCQs.")
	state := fakeState{
		"source_files_json": `[{"blob_uri":"s3://b/job/source","mime_type":"application/pdf","role":"source"},` +
			`{"blob_uri":"s3://b/job/rubric","mime_type":"text/markdown","role":"rubric"}]`,
		"source_blob_uri":      "s3://b/job/source",
		"source_mime_type":     "application/pdf",
		"source_material_text": "rendered material",
		"grounding_mode":       "starting_point",
	}

	if err := applyGroundingFromState(state, req); err != nil {
		t.Fatalf("applyGroundingFromState: %v", err)
	}
	for i, p := range req.Contents[0].Parts {
		if p.FileData != nil || p.InlineData != nil {
			t.Fatalf("part %d is a blob (%+v); the canonical gateway contract cannot carry it", i, p)
		}
	}
}

// Declaring material but supplying none must FAIL, not quietly generate
// ungrounded questions the author believes came from their document.
func TestApplyGrounding_DeclaredFileListWithoutTextFails(t *testing.T) {
	req := userReq("Generate MCQs.")
	state := fakeState{
		"source_files_json": `[{"blob_uri":"s3://b/job/source","mime_type":"application/pdf","role":"source"}]`,
	}

	err := applyGroundingFromState(state, req)
	if !errors.Is(err, ErrNoSourceMaterialText) {
		t.Fatalf("err = %v; want ErrNoSourceMaterialText", err)
	}
	if len(req.Contents[0].Parts) != 1 {
		t.Errorf("parts = %d; want the request untouched on failure", len(req.Contents[0].Parts))
	}
}

func TestApplyGrounding_SingleBlobWithoutTextFails(t *testing.T) {
	req := userReq("Generate MCQs.")
	state := fakeState{
		"source_blob_uri":  "s3://b/job/source",
		"source_mime_type": "image/png",
	}

	// An image-only upload has no text layer by design — it cannot be
	// text-grounded, so it fails truthfully rather than inventing questions.
	if err := applyGroundingFromState(state, req); !errors.Is(err, ErrNoSourceMaterialText) {
		t.Fatalf("err = %v; want ErrNoSourceMaterialText", err)
	}
}

func TestApplyGrounding_NoMaterialIsNoOp(t *testing.T) {
	req := userReq("x")
	if err := applyGroundingFromState(fakeState{}, req); err != nil {
		t.Fatalf("err = %v; want nil for a job with no material", err)
	}
	if len(req.Contents[0].Parts) != 1 {
		t.Errorf("no-op expected; parts = %d, want 1", len(req.Contents[0].Parts))
	}
}

func TestApplyGrounding_WhitespaceTextIsAbsent(t *testing.T) {
	req := userReq("x")
	if err := applyGroundingFromState(fakeState{"source_material_text": "   \n\t "}, req); err != nil {
		t.Fatalf("err = %v; want nil (whitespace is not material)", err)
	}
	if len(req.Contents[0].Parts) != 1 {
		t.Errorf("no-op expected; parts = %d, want 1", len(req.Contents[0].Parts))
	}
}

func TestApplyGrounding_NoUserContentAppendsFresh(t *testing.T) {
	req := &adkmodel.LLMRequest{Contents: nil}
	state := fakeState{"source_material_text": "material"}

	if err := applyGroundingFromState(state, req); err != nil {
		t.Fatalf("applyGroundingFromState: %v", err)
	}

	if len(req.Contents) != 1 || req.Contents[0].Role != "user" {
		t.Fatalf("expected a fresh user content; got %+v", req.Contents)
	}
	if !strings.Contains(req.Contents[0].Parts[0].Text, "material") {
		t.Error("fresh content missing the grounding text")
	}
}

func TestApplyGrounding_PrefersLastUserContent(t *testing.T) {
	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "older"}}},
			{Role: "model", Parts: []*genai.Part{{Text: "reply"}}},
			{Role: "user", Parts: []*genai.Part{{Text: "newest"}}},
		},
	}
	if err := applyGroundingFromState(fakeState{"source_material_text": "material"}, req); err != nil {
		t.Fatalf("applyGroundingFromState: %v", err)
	}

	// The material rides with the NEWEST user turn, not the older one or the model turn.
	if n := len(req.Contents[2].Parts); n != 2 {
		t.Fatalf("newest user content parts = %d; want 2", n)
	}
	if n := len(req.Contents[0].Parts); n != 1 {
		t.Errorf("older user turn parts = %d; want 1", n)
	}
	if n := len(req.Contents[1].Parts); n != 1 {
		t.Errorf("model turn parts = %d; want 1", n)
	}
}

func TestApplyGrounding_NilReqSafe(t *testing.T) {
	if err := applyGroundingFromState(fakeState{"source_material_text": "m"}, nil); err != nil {
		t.Fatalf("err = %v; want nil on a nil request", err)
	}
}

func TestApplyGrounding_StateErrorIsNoOp(t *testing.T) {
	req := userReq("x")
	// Get errors → readState "" → no material declared → no-op, not a refusal.
	if err := applyGroundingFromState(errState{}, req); err != nil {
		t.Fatalf("err = %v; want nil", err)
	}
	if len(req.Contents[0].Parts) != 1 {
		t.Errorf("state-error must be a no-op; parts = %d, want 1", len(req.Contents[0].Parts))
	}
}

func TestApplyGrounding_NonStringTextIsNoOp(t *testing.T) {
	req := userReq("x")
	if err := applyGroundingFromState(fakeState{"source_material_text": 123}, req); err != nil {
		t.Fatalf("err = %v; want nil", err)
	}
	if len(req.Contents[0].Parts) != 1 {
		t.Errorf("non-string text must be a no-op; parts = %d, want 1", len(req.Contents[0].Parts))
	}
}

func TestApplyGrounding_SkipsNilContent(t *testing.T) {
	// A nil entry at the TAIL must be skipped (the loop scans from the end), not
	// panic; the text lands on the preceding real user content.
	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "x"}}}, nil},
	}
	if err := applyGroundingFromState(fakeState{"source_material_text": "material"}, req); err != nil {
		t.Fatalf("applyGroundingFromState: %v", err)
	}
	if len(req.Contents[0].Parts) != 2 {
		t.Errorf("text should land on the real user content; parts = %d, want 2", len(req.Contents[0].Parts))
	}
}

func TestNew_BuildsPlugin(t *testing.T) {
	p, err := New("qgen_question")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if p == nil {
		t.Fatal("New returned nil plugin")
	}
}
