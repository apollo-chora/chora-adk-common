package agentdispatch

import (
	"testing"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

func ev(author, text string, partial bool) *session.Event {
	return &session.Event{
		Author: author,
		LLMResponse: model.LLMResponse{
			Partial: partial,
			Content: &genai.Content{Parts: []*genai.Part{{Text: text}}},
		},
	}
}

// The HTTP executor picked the terminal answer as "the LAST non-partial event
// from the terminal author, with non-empty text". The in-process runner has to
// pick the SAME event or the two transports return different answers from the
// same agent, and ADR-253 D6 rollback stops being a rollback.

func TestTerminalTextTakesTheLastNonPartialEventFromTheAuthor(t *testing.T) {
	events := []*session.Event{
		ev("oe_evaluator", `{"partial":1}`, true),
		ev("oe_evaluator", `{"comment":"first"}`, false),
		ev("oe_evaluator", `{"comment":"final"}`, false),
	}
	got, err := TerminalText(events, "oe_evaluator")
	if err != nil {
		t.Fatalf("TerminalText: %v", err)
	}
	if got != `{"comment":"final"}` {
		t.Errorf("got %q", got)
	}
}

func TestTerminalTextIgnoresOtherAuthors(t *testing.T) {
	events := []*session.Event{
		ev("oe_evaluator", `{"comment":"mine"}`, false),
		ev("some_tool", `{"comment":"not mine"}`, false),
	}
	got, err := TerminalText(events, "oe_evaluator")
	if err != nil {
		t.Fatalf("TerminalText: %v", err)
	}
	if got != `{"comment":"mine"}` {
		t.Errorf("got %q", got)
	}
}

func TestTerminalTextFailsLoudWhenTheAgentSaidNothing(t *testing.T) {
	// The HTTP executor raised here too. Returning "" would publish an empty
	// completion and the orchestrator would grade the answer as zero.
	if _, err := TerminalText([]*session.Event{
		ev("oe_evaluator", "", false),
	}, "oe_evaluator"); err == nil {
		t.Fatal("accepted an empty terminal event")
	}
}

func TestTerminalTextConcatenatesMultipartText(t *testing.T) {
	e := &session.Event{Author: "oe_evaluator", LLMResponse: model.LLMResponse{
		Content: &genai.Content{Parts: []*genai.Part{{Text: `{"a":`}, {Text: `1}`}}},
	}}
	got, err := TerminalText([]*session.Event{e}, "oe_evaluator")
	if err != nil {
		t.Fatalf("TerminalText: %v", err)
	}
	if got != `{"a":1}` {
		t.Errorf("got %q", got)
	}
}

func TestTerminalAuthorMatchesTheDeployedAgentNames(t *testing.T) {
	for role, want := range map[string]string{
		"oe_evaluate": "oe_evaluator",
		"oe_moderate": "oe_moderator",
	} {
		if got := TerminalAuthor(role); got != want {
			t.Errorf("TerminalAuthor(%q) = %q, want %q", role, got, want)
		}
	}
}
