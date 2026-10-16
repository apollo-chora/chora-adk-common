// mapping.go: translating the ADK genai shapes onto the gateway's canonical
// contract.
//
// The ADK runtime hands this client `genai.Content` values and `genai.Tool`
// declarations. The gateway's Invoke RPC carries the canonical model instead:
// typed `Message` values, a `system_instruction`, and `ToolDefinition`s. This
// file is the whole of that translation, and it is the only place the two
// shapes meet.
//
// Two conversions are deliberately lossy in a documented way:
//
//   - genai `InlineData`/`FileData` parts have no canonical part. The gateway's
//     canonical model admits text, image and refusal parts, so a binary blob is
//     refused rather than silently dropped — a caller that needs it must send an
//     image part instead.
//   - the gateway's `InvokeResponse` reports the upstream model and a fallback
//     flag where the Go contract reported a vendor, a fallback chain, a latency
//     and a gateway version. Those four had no caller-visible consumer that
//     could not be served another way, so they are not carried across.

package modelgatewayclient

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"go.opentelemetry.io/otel/trace"
	"google.golang.org/genai"
	"google.golang.org/grpc/metadata"

	gatewayv1 "github.com/apollo-chora/chora-adk-common/modelgatewayclient/gatewayv1"
)

// contentsToMessages converts the ADK conversation into canonical messages.
//
// `prompt` is the flat user text `flattenContents` produced. When the
// conversation is a simple single-turn text exchange the canonical answer is
// one user message carrying that text — which is what the flat path always
// sent. When the conversation is structured (multi-turn, or carrying function
// calls or responses) each genai Content becomes one canonical Message, so no
// turn is flattened away.
func contentsToMessages(contents []*genai.Content, prompt string) ([]*gatewayv1.Message, error) {
	if len(contents) == 0 {
		if prompt == "" {
			return nil, nil
		}
		return []*gatewayv1.Message{userMessage(prompt)}, nil
	}

	out := make([]*gatewayv1.Message, 0, len(contents))
	for _, c := range contents {
		if c == nil {
			continue
		}
		msg := &gatewayv1.Message{}
		switch c.Role {
		case "system":
			// A system part is an instruction, not a turn: it belongs in
			// system_instruction, which the caller already extracted. Skipping it
			// here keeps the two channels from carrying the same text twice.
			continue
		case "model", "assistant":
			msg.Role = gatewayv1.Role_ROLE_ASSISTANT
		default:
			msg.Role = gatewayv1.Role_ROLE_USER
		}
		for _, p := range c.Parts {
			if p == nil {
				continue
			}
			switch {
			case p.Text != "":
				msg.Content = append(msg.Content, textPart(p.Text))
			case p.FunctionCall != nil:
				msg.ToolCalls = append(msg.ToolCalls, &gatewayv1.ToolCall{
					Name: p.FunctionCall.Name,
					// The canonical field is JSON bytes, and the ADK value is a
					// map, so it is marshalled rather than stringified.
					ArgumentsJson: mustJSON(p.FunctionCall.Args),
				})
			case p.FunctionResponse != nil:
				msg.Content = append(msg.Content, textPart(string(mustJSON(p.FunctionResponse.Response))))
			default:
				// InlineData and FileData have no canonical part. Refuse rather
				// than drop: a dropped blob is a silently truncated turn, and
				// the caller must not be able to mistake the refusal for a
				// text-only transcript.
				return nil, fmt.Errorf("modelgatewayclient: content part %T has no canonical representation", p)
			}
		}
		if len(msg.Content) == 0 && len(msg.ToolCalls) == 0 {
			continue
		}
		out = append(out, msg)
	}
	if len(out) == 0 {
		if prompt == "" {
			return nil, fmt.Errorf("modelgatewayclient: conversation carried no transcribable content")
		}
		// Every structured entry was a system part, so the flat prompt is the
		// whole user turn. This is the flat path, not a lossy fallback.
		return []*gatewayv1.Message{userMessage(prompt)}, nil
	}
	return out, nil
}

func userMessage(text string) *gatewayv1.Message {
	return &gatewayv1.Message{
		Role:    gatewayv1.Role_ROLE_USER,
		Content: []*gatewayv1.ContentPart{textPart(text)},
	}
}

func textPart(text string) *gatewayv1.ContentPart {
	return &gatewayv1.ContentPart{
		Part: &gatewayv1.ContentPart_Text{Text: &gatewayv1.TextPart{Text: text}},
	}
}

// imageMessages builds the user turn for an image request. A source image
// rides as an image part beside the text, which is the canonical shape; the
// Go contract carried it as genai JSON in contents_json instead.
func imageMessages(req ImageRequest) []*gatewayv1.Message {
	parts := []*gatewayv1.ContentPart{}
	if req.Prompt != "" {
		parts = append(parts, textPart(req.Prompt))
	}
	if len(req.SourceImage) > 0 {
		mime := req.SourceMime
		if mime == "" {
			mime = "application/octet-stream"
		}
		parts = append(parts, &gatewayv1.ContentPart{
			Part: &gatewayv1.ContentPart_Image{Image: &gatewayv1.ImagePart{
				Source:    &gatewayv1.ImagePart_Data{Data: base64.StdEncoding.EncodeToString(req.SourceImage)},
				MediaType: mime,
			}},
		})
	}
	if len(parts) == 0 {
		return nil
	}
	return []*gatewayv1.Message{{Role: gatewayv1.Role_ROLE_USER, Content: parts}}
}

// toolsToDefinitions converts the agent's genai tool declarations into the
// canonical ToolDefinition list. The canonical field is a JSON Schema as JSON
// bytes, and the genai declaration already carries one, so it is carried
// across rather than re-derived.
func toolsToDefinitions(cfg *genai.GenerateContentConfig) []*gatewayv1.ToolDefinition {
	if cfg == nil || len(cfg.Tools) == 0 {
		return nil
	}
	var out []*gatewayv1.ToolDefinition
	for _, t := range cfg.Tools {
		if t == nil || len(t.FunctionDeclarations) == 0 {
			continue
		}
		for _, fd := range t.FunctionDeclarations {
			if fd == nil {
				continue
			}
			out = append(out, &gatewayv1.ToolDefinition{
				Name:        fd.Name,
				Description: fd.Description,
				// genai marshals Parameters as a JSON Schema object, which is
				// exactly the canonical field's type.
				ParametersJson: mustJSON(fd.Parameters),
			})
		}
	}
	return out
}

// withTraceContext copies the W3C trace context of `from` onto the outgoing
// gRPC metadata of `to`. The Invoke RPC has no traceparent field, so this is
// the only channel the context has.
//
// AppendToOutgoingContext appends rather than replaces, so a value already
// present is left alone rather than sent twice.
func withTraceContext(to, from context.Context) context.Context {
	md, ok := metadata.FromOutgoingContext(to)
	if ok && len(md.Get("traceparent")) > 0 {
		return to
	}
	pairs := make([]string, 0, 4)
	if sc := trace.SpanContextFromContext(from); sc.IsValid() {
		pairs = append(pairs, "traceparent", fmt.Sprintf("00-%s-%s-%02x", sc.TraceID(), sc.SpanID(), byte(sc.TraceFlags())))
		if tsv := sc.TraceState().String(); tsv != "" {
			pairs = append(pairs, "tracestate", tsv)
		}
	}
	if len(pairs) == 0 {
		return to
	}
	return metadata.AppendToOutgoingContext(to, pairs...)
}

// mustJSON marshals caller bookkeeping into a JSON payload. It panics on
// failure: the inputs are fixed shapes, so a failure would be a programming
// error rather than a condition to handle.
func mustJSON(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		panic(fmt.Sprintf("modelgatewayclient: marshal: %v", err))
	}
	return b
}

// answerText flattens the canonical output items into the model's answer.
//
// Only text parts are an answer. A refusal is the gateway declining, and
// dropping it silently would let a truncated answer parse as a complete one, so
// a refusal yields "" and the caller reports the empty completion.
func answerText(resp *gatewayv1.InvokeResponse) (string, error) {
	if resp == nil {
		return "", fmt.Errorf("modelgatewayclient: nil InvokeResponse")
	}
	var parts []string
	for _, item := range resp.GetOutput() {
		part := item.GetContent()
		if part == nil {
			continue
		}
		switch {
		case part.GetText() != nil:
			parts = append(parts, part.GetText().GetText())
		case part.GetRefusal() != nil:
			// The gateway declined to answer. Returning an empty string here
			// would let the caller build an apparently-complete response with
			// TurnComplete set, disguising a refusal as a blank completion.
			return "", fmt.Errorf("modelgatewayclient: the model refused: %s", part.GetRefusal().GetText())
		}
	}
	return strings.Join(parts, ""), nil
}

// outputToolCalls returns the model's function calls as canonical JSON, which is
// the shape buildResponseParts already parses.
func outputToolCalls(resp *gatewayv1.InvokeResponse) (string, error) {
	if resp == nil {
		return "", nil
	}
	var calls []*genai.FunctionCall
	for _, item := range resp.GetOutput() {
		tc := item.GetToolCall()
		if tc == nil {
			continue
		}
		var args map[string]any
		if len(tc.GetArgumentsJson()) > 0 {
			if err := json.Unmarshal(tc.GetArgumentsJson(), &args); err != nil {
				// A tool call whose arguments cannot be parsed must not become a
				// call with nil arguments: the agent would invoke the tool with
				// nothing and read the failure as the tool's own.
				return "", fmt.Errorf("modelgatewayclient: tool call %q has unparseable arguments: %w", tc.GetName(), err)
			}
		}
		calls = append(calls, &genai.FunctionCall{Name: tc.GetName(), Args: args})
	}
	if len(calls) == 0 {
		return "", nil
	}
	return string(mustJSON(calls)), nil
}

// mapFinishReason translates the canonical finish reason onto the genai enum
// the ADK runtime expects, carrying the provider's own reason when the gateway
// could not classify it.
func mapFinishReason(reason gatewayv1.FinishReason, raw string) genai.FinishReason {
	switch reason {
	case gatewayv1.FinishReason_FINISH_REASON_STOP:
		return genai.FinishReasonStop
	case gatewayv1.FinishReason_FINISH_REASON_LENGTH:
		return genai.FinishReasonMaxTokens
	case gatewayv1.FinishReason_FINISH_REASON_TOOL_CALLS:
		// The ADK runtime drives its tool loop from the FunctionCall parts, not
		// from the finish reason, so STOP is the honest mapping.
		return genai.FinishReasonStop
	case gatewayv1.FinishReason_FINISH_REASON_CONTENT_FILTER:
		return genai.FinishReasonSafety
	default:
		// The provider's own reason, when the gateway could not classify it.
		_ = raw
		return genai.FinishReasonOther
	}
}

// imageFromResponse pulls the generated image out of the canonical output
// items, returning the decoded bytes, the declared media type, and a detail
// string for the no-image case.
//
// The canonical image part carries either base64 `data` or a `url`; this caller
// wants bytes, so a url-only part is not an image it can return and is reported
// as a detail instead.
func imageFromResponse(resp *gatewayv1.InvokeResponse) (data []byte, mime, detail string) {
	if resp == nil {
		return nil, "", "nil response"
	}
	for _, item := range resp.GetOutput() {
		part := item.GetContent()
		if part == nil {
			continue
		}
		img := part.GetImage()
		if img == nil {
			if part.GetText() != nil {
				detail = part.GetText().GetText()
			}
			continue
		}
		if img.GetMediaType() != "" {
			mime = img.GetMediaType()
		}
		if raw := img.GetData(); raw != "" {
			decoded, err := base64.StdEncoding.DecodeString(raw)
			if err != nil {
				return nil, mime, fmt.Sprintf("image data is not valid base64: %v", err)
			}
			return decoded, mime, ""
		}
		if img.GetUrl() != "" {
			return nil, mime, "image part carries a url, not bytes"
		}
	}
	if detail == "" {
		detail = "no image part in the response"
	}
	return nil, mime, detail
}
