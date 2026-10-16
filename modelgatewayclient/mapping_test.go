package modelgatewayclient

import (
	"context"
	"strings"
	"testing"

	"google.golang.org/genai"
	"google.golang.org/grpc/metadata"

	gatewayv1 "github.com/apollo-chora/chora-adk-common/modelgatewayclient/gatewayv1"
)

// An unsupported part must be refused, not dropped — a dropped blob is a
// silently truncated turn.
func TestContentsToMessages_RefusesUnsupportedParts(t *testing.T) {
	contents := []*genai.Content{{
		Role:  "user",
		Parts: []*genai.Part{{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte("x")}}},
	}}
	if _, err := contentsToMessages(contents, ""); err == nil {
		t.Fatal("expected an error for an InlineData part")
	}
}

// A conversation of nothing but system parts is the flat path, not a lossy
// fallback: the flat prompt is the whole user turn.
func TestContentsToMessages_OnlySystemPartsUsesTheFlatPrompt(t *testing.T) {
	contents := []*genai.Content{{Role: "system", Parts: []*genai.Part{{Text: "instructions"}}}}
	msgs, err := contentsToMessages(contents, "the user turn")
	if err != nil {
		t.Fatalf("contentsToMessages: %v", err)
	}
	if len(msgs) != 1 || msgs[0].GetContent()[0].GetText().GetText() != "the user turn" {
		t.Fatalf("messages: %v", msgs)
	}
}

// A multi-turn conversation must survive with every turn present.
func TestContentsToMessages_KeepsEveryTurn(t *testing.T) {
	contents := []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "first"}}},
		{Role: "model", Parts: []*genai.Part{{Text: "second"}}},
		{Role: "user", Parts: []*genai.Part{{Text: "third"}}},
	}
	msgs, err := contentsToMessages(contents, "")
	if err != nil {
		t.Fatalf("contentsToMessages: %v", err)
	}
	if len(msgs) != 3 {
		t.Fatalf("turns: got %d, want 3", len(msgs))
	}
	if msgs[1].GetRole() != gatewayv1.Role_ROLE_ASSISTANT {
		t.Errorf("assistant turn role: got %v", msgs[1].GetRole())
	}
}

// A refusal is an error, not an empty completion the caller would read as a
// completed turn.
func TestAnswerText_RefusalIsAnError(t *testing.T) {
	resp := &gatewayv1.InvokeResponse{Output: []*gatewayv1.OutputItem{{
		Item: &gatewayv1.OutputItem_Content{Content: &gatewayv1.ContentPart{
			Part: &gatewayv1.ContentPart_Refusal{Refusal: &gatewayv1.RefusalPart{Text: "not allowed"}},
		}},
	}}}
	if _, err := answerText(resp); err == nil || !strings.Contains(err.Error(), "not allowed") {
		t.Fatalf("answerText: err = %v, want a refusal naming the text", err)
	}
}

// A tool call whose arguments cannot be parsed must fail, not become a call with
// nil arguments.
func TestOutputToolCalls_RefusesUnparseableArguments(t *testing.T) {
	resp := &gatewayv1.InvokeResponse{Output: []*gatewayv1.OutputItem{{
		Item: &gatewayv1.OutputItem_ToolCall{ToolCall: &gatewayv1.ToolCall{
			Name: "lookup", ArgumentsJson: []byte("{not json"),
		}},
	}}}
	if _, err := outputToolCalls(resp); err == nil {
		t.Fatal("expected an error for unparseable tool-call arguments")
	}
}

// Generation settings the canonical config has no field for must be refused
// rather than silently changing what the model is asked for.
func TestEncodeGenerationConfig_RefusesUnsupportedSettings(t *testing.T) {
	topK := float32(40)
	if _, err := encodeGenerationConfig(&genai.GenerateContentConfig{TopK: &topK}); err == nil {
		t.Fatal("expected an error for top_k")
	}
	mime := "application/json"
	if _, err := encodeGenerationConfig(&genai.GenerateContentConfig{ResponseMIMEType: mime}); err == nil {
		t.Fatal("expected an error for response_mime_type")
	}
}

// An existing outgoing traceparent is authoritative: it must not be duplicated,
// and the tracestate must still be added when it is missing.
func TestWithTraceContext_KeepsAnExistingParentAndAddsState(t *testing.T) {
	ctx := metadata.AppendToOutgoingContext(context.Background(), "traceparent", "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01")
	out := withTraceContext(ctx, context.Background())
	md, _ := metadata.FromOutgoingContext(out)
	if got := md.Get("traceparent"); len(got) != 1 {
		t.Fatalf("traceparent: got %v, want exactly one value", got)
	}
}
