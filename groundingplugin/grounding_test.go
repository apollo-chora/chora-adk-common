package groundingplugin

import (
	"errors"
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

func TestApplyGrounding_InjectsFileDataPart(t *testing.T) {
	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "Generate MCQs."}}}},
	}
	state := fakeState{
		"source_blob_uri":  "gs://b/job/material.pdf",
		"source_mime_type": "application/pdf",
	}

	applyGroundingFromState(state, req)

	parts := req.Contents[0].Parts
	if len(parts) != 2 {
		t.Fatalf("parts = %d; want 2 (text + fileData)", len(parts))
	}
	fd := parts[1].FileData
	if fd == nil {
		t.Fatal("fileData part not injected")
	}
	if fd.FileURI != "gs://b/job/material.pdf" || fd.MIMEType != "application/pdf" {
		t.Errorf("fileData = {%q, %q}; want gs://b/job/material.pdf application/pdf", fd.FileURI, fd.MIMEType)
	}
}

func TestApplyGrounding_NoBlobIsNoOp(t *testing.T) {
	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "x"}}}},
	}
	applyGroundingFromState(fakeState{}, req) // no source_blob_uri
	if len(req.Contents[0].Parts) != 1 {
		t.Errorf("no-op expected for empty source_blob_uri; parts = %d, want 1", len(req.Contents[0].Parts))
	}
}

func TestApplyGrounding_NoUserContentAppendsFresh(t *testing.T) {
	req := &adkmodel.LLMRequest{Contents: nil}
	state := fakeState{"source_blob_uri": "gs://b/m.png", "source_mime_type": "image/png"}

	applyGroundingFromState(state, req)

	if len(req.Contents) != 1 || req.Contents[0].Role != "user" {
		t.Fatalf("expected a fresh user content; got %+v", req.Contents)
	}
	if req.Contents[0].Parts[0].FileData == nil {
		t.Error("fresh content missing fileData part")
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
	applyGroundingFromState(fakeState{"source_blob_uri": "gs://b/m.pdf"}, req)

	// The blob rides with the NEWEST user turn, not the older one or the model turn.
	if n := len(req.Contents[2].Parts); n != 2 {
		t.Fatalf("newest user content parts = %d; want 2", n)
	}
	if req.Contents[0].Parts[len(req.Contents[0].Parts)-1].FileData != nil {
		t.Error("fileData wrongly attached to the older user turn")
	}
	if req.Contents[1].Parts[len(req.Contents[1].Parts)-1].FileData != nil {
		t.Error("fileData wrongly attached to the model turn")
	}
}

func TestApplyGrounding_NilReqSafe(t *testing.T) {
	// Must not panic.
	applyGroundingFromState(fakeState{"source_blob_uri": "gs://x"}, nil)
}

func TestApplyGrounding_StateErrorIsNoOp(t *testing.T) {
	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "x"}}}},
	}
	applyGroundingFromState(errState{}, req) // Get errors → readState "" → no-op
	if len(req.Contents[0].Parts) != 1 {
		t.Errorf("state-error must be a no-op; parts = %d, want 1", len(req.Contents[0].Parts))
	}
}

func TestApplyGrounding_NonStringBlobIsNoOp(t *testing.T) {
	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "x"}}}},
	}
	applyGroundingFromState(fakeState{"source_blob_uri": 123}, req) // non-string → "" → no-op
	if len(req.Contents[0].Parts) != 1 {
		t.Errorf("non-string blob must be a no-op; parts = %d, want 1", len(req.Contents[0].Parts))
	}
}

func TestApplyGrounding_SkipsNilContent(t *testing.T) {
	// A nil entry at the TAIL must be skipped (the loop scans from the end), not
	// panic; the blob lands on the preceding real user content.
	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "x"}}}, nil},
	}
	applyGroundingFromState(fakeState{"source_blob_uri": "gs://b/m.pdf"}, req)
	if len(req.Contents[0].Parts) != 2 {
		t.Errorf("blob should land on the real user content; parts = %d, want 2", len(req.Contents[0].Parts))
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

func TestApplyGrounding_AttachesEveryEntryOfTheMultiFileList(t *testing.T) {
	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "Generate MCQs."}}}},
	}
	state := fakeState{
		"source_files_json": `[{"blob_uri":"s3://b/job/source","mime_type":"application/pdf","role":"source"},` +
			`{"blob_uri":"s3://b/job/rubric","mime_type":"text/markdown","role":"rubric"}]`,
	}

	applyGroundingFromState(state, req)

	parts := req.Contents[0].Parts
	if len(parts) != 3 {
		t.Fatalf("parts = %d; want 3 (text + 2 fileData)", len(parts))
	}
	if got := parts[1].FileData.FileURI; got != "s3://b/job/source" {
		t.Errorf("first file = %q; want s3://b/job/source", got)
	}
	if got := parts[2].FileData.FileURI; got != "s3://b/job/rubric" {
		t.Errorf("second file = %q; want s3://b/job/rubric", got)
	}
}

func TestApplyGrounding_MalformedFileListFallsBackToTheSingleBlob(t *testing.T) {
	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "Generate MCQs."}}}},
	}
	state := fakeState{
		"source_files_json": "not json",
		"source_blob_uri":   "s3://b/job/source",
		"source_mime_type":  "application/pdf",
	}

	applyGroundingFromState(state, req)

	parts := req.Contents[0].Parts
	if len(parts) != 2 || parts[1].FileData == nil || parts[1].FileData.FileURI != "s3://b/job/source" {
		t.Fatalf("malformed list must fall back to the single blob; parts = %+v", parts)
	}
}
