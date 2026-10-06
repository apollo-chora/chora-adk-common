package agentdispatch

import (
	"context"
	"fmt"
	"strings"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// triggerText is the user message. The real prompt is injected into the agent's
// [TASK] block through session state, so this is purely a "start now" signal.
//
// ⚠ It is "BEGIN" and must stay "BEGIN". The literal "go" was tried and Gemini
// read it as a topic hint (the Go language / the shell command), which polluted
// generated content. Same string the HTTP path sends, so the two transports
// present the model with identical input.
const triggerText = "BEGIN"

// TerminalAuthor names the agent whose final non-partial event carries the
// structured output for a role. Mirrors the HTTP executor's
// _terminal_author_for; these are ADK agent names, not role names, and the two
// deliberately differ (role oe_evaluate is served by agent oe_evaluator).
func TerminalAuthor(agentRole string) string {
	switch agentRole {
	case "oe_evaluate":
		return "oe_evaluator"
	case "oe_moderate":
		return "oe_moderator"
	case "qgen_question":
		// Trimmed to a single generation sub-agent 2026-06-01; generation is
		// terminal and emits the candidate directly (the HTTP-era role name
		// the kennel's executor keys on).
		return "qgen_question_generation"
	case "qgen_critic", "qgen_critique":
		return "qgen_critic"
	}
	// qgen_generate (ADR-254 lane role, mode generate | compose) and every
	// other lane role: the LAST text event wins; generate's last text event
	// is the generation sub-agent's candidate, compose's is the finaliser's
	// envelope.
	return ""
}

// TerminalText picks the agent's answer out of an event stream.
//
// The selection rule is the HTTP executor's, kept identical on purpose: the
// LAST non-partial event from the terminal author with non-empty text. A
// different rule here would make the two transports return different answers
// from the same agent, and ADR-253 D6 rollback would stop being a rollback.
//
// Fails loud on nothing-said. Returning "" would publish an empty completion,
// and the orchestrator would score the learner's answer as zero on what is
// actually an agent fault.
func TerminalText(events []*session.Event, terminalAuthor string) (string, error) {
	terminal := ""
	for _, e := range events {
		if e == nil || e.Content == nil {
			continue
		}
		if terminalAuthor != "" && e.Author != terminalAuthor {
			continue
		}
		if e.Partial {
			continue
		}
		var sb strings.Builder
		for _, p := range e.Content.Parts {
			if p != nil {
				sb.WriteString(p.Text)
			}
		}
		if text := sb.String(); text != "" {
			terminal = text
		}
	}
	if terminal == "" {
		return "", fmt.Errorf(
			"agentdispatch: no terminal event from author %q; the agent produced "+
				"no structured output", terminalAuthor)
	}
	return terminal, nil
}

// RunPolicy is the per-role session and user-message policy (see
// ServeConfig.SessionKey / UserMessage). The zero value is the stateless
// per-dispatch policy every crew ran on before companion_chat.
type RunPolicy struct {
	SessionKey  func(req Request) (userID, sessionID string, ok bool)
	UserMessage func(req Request) string
}

// RunnerAgentRunWithPolicy builds the AgentRun that executes the dispatched
// work IN-PROCESS through the ADK runner, under a per-role RunPolicy. This is
// the native subscriber ADR-253 D7 rules for: the same root agent and the same
// plugin chain the HTTP launcher serves, reached without an HTTP hop, rather
// than a shim that posts to the agent's own local surface.
//
// The session service is passed alongside the runner rather than read from it:
// runner.Runner keeps its session service unexported, and the caller builds
// both from the same launcher config anyway. They MUST be the same instance —
// a runner looking at a different store would find no session and no state.
//
// Stateless roles get a fresh per-dispatch session and the user message is
// handed to the runner, which appends it. A conversational role (persistent
// session) has its turn appended HERE, once per dispatch, under the dispatch's
// own invocation id and carrying the request's state delta; the runner then
// starts with no message and answers the trailing user turn. That is what makes
// a REDELIVERY safe on a conversation that outlives the dispatch: the retry
// finds its turn already recorded and resumes instead of appending the
// learner's message again (seen live 2026-08-22: five attempts of one turn
// left five copies of the message in the dialogue, which the model then
// quoted back five times).
func RunnerAgentRunWithPolicy(r *runner.Runner, sessions session.Service, appName string, policy RunPolicy) AgentRun {
	return func(ctx context.Context, req Request) (string, error) {
		userID, sessionID, persistent := resolveSessionKey(policy, req)
		text := triggerText
		if policy.UserMessage != nil {
			if t := strings.TrimSpace(policy.UserMessage(req)); t != "" {
				text = t
			}
		}
		var msg *genai.Content
		if persistent {
			if _, err := prepareConversationTurn(ctx, sessions, appName, req, userID, sessionID, text); err != nil {
				return "", err
			}
		} else {
			if err := prepareStatelessSession(ctx, sessions, appName, req, userID, sessionID); err != nil {
				return "", err
			}
			msg = &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{Text: text}},
			}
		}
		var events []*session.Event
		for e, err := range r.Run(ctx, userID, sessionID, msg, agent.RunConfig{}) {
			if err != nil {
				// Permanent faults (the gateway's FAILED_PRECONDITION tokens, an
				// agent's own Permanent) keep their classification through the
				// wrap so the handler can report them at once (ADR-254 D5/D6).
				return "", classifyRunError(fmt.Errorf("agentdispatch: run %q: %w", req.ExecutionID, err))
			}
			events = append(events, e)
		}
		author := TerminalAuthor(req.AgentRole)
		return TerminalText(events, author)
	}
}

// prepareSession creates this dispatch's session, clearing any stale one first.
//
// ⚠ The clear is not tidiness, it is what makes a RETRY possible. The event
// bus is at-least-once and the handler NACKs a transient agent failure, so a
// redelivery arrives with the same execution id and therefore the same session
// id. Live on 2026-08-20: attempt 1 failed on a real cause, and every retry
// then died on "session dispatch:<id> already exists" instead — so a lane that
// hit one transient error could never recover, and the message dead-lettered
// five attempts later reporting the wrong reason entirely.
//
// Deleting rather than reusing is deliberate. The OE crews are stateless and
// single-turn per dispatch, so a retry must see a CLEAN session: reusing one
// that already holds the previous attempt's events would feed the model a
// conversation it never had.
func prepareSession(
	ctx context.Context, sessions session.Service, appName string, req Request,
) error {
	return prepareStatelessSession(ctx, sessions, appName, req, userIDFor(req), sessionIDFor(req))
}

// resolveSessionKey applies the policy: a conversational role names the
// session itself; everything else keys it on the dispatch.
func resolveSessionKey(policy RunPolicy, req Request) (userID, sessionID string, persistent bool) {
	if policy.SessionKey != nil {
		if u, s, ok := policy.SessionKey(req); ok && strings.TrimSpace(u) != "" && strings.TrimSpace(s) != "" {
			return strings.TrimSpace(u), strings.TrimSpace(s), true
		}
	}
	return userIDFor(req), sessionIDFor(req), false
}

// prepareStatelessSession clears any stale session and creates a fresh one
// carrying the request's state (see prepareSession for why the clear matters).
func prepareStatelessSession(
	ctx context.Context, sessions session.Service, appName string, req Request,
	userID, sessionID string,
) error {
	state, err := req.SessionState()
	if err != nil {
		return err
	}
	// Best-effort: on the first delivery there is nothing to delete, and the
	// service reports that in its own way. The create below is the real gate.
	_ = sessions.Delete(ctx, &session.DeleteRequest{
		AppName: appName, UserID: userID, SessionID: sessionID,
	})
	if _, err := sessions.Create(ctx, &session.CreateRequest{
		AppName: appName, UserID: userID, SessionID: sessionID, State: state,
	}); err != nil {
		return fmt.Errorf("agentdispatch: create session for %q: %w", req.ExecutionID, err)
	}
	return nil
}

// dispatchInvocationID is the invocation id a dispatch's turn is recorded
// under in a persistent conversation ("dispatch:<execution id>"); a
// redelivery carries the same execution id and finds it.
func dispatchInvocationID(req Request) string { return sessionIDFor(req) }

// prepareConversationTurn readies a persistent conversation (companion_chat,
// ADR-254 D6) for this dispatch. The session is the CONVERSATION: created on
// first sight and never deleted here, so the event history, which is the
// dialogue, survives across dispatches, pods and redeliveries.
//
// The turn itself is ONE user event under the dispatch's invocation id: the
// learner's text (or the role's trigger) as content, the request's state as
// the event's state delta so the turn's inputs (message, tier, memory, growth
// edges, dispatch key) are current. When that event is already in the
// conversation, this is a redelivery: nothing is appended and resumed reports
// true, so the caller runs the agent on the history as it stands.
func prepareConversationTurn(
	ctx context.Context, sessions session.Service, appName string, req Request,
	userID, sessionID, text string,
) (resumed bool, err error) {
	state, err := req.SessionState()
	if err != nil {
		return false, err
	}
	invocationID := dispatchInvocationID(req)
	var sess session.Session
	got, err := sessions.Get(ctx, &session.GetRequest{AppName: appName, UserID: userID, SessionID: sessionID})
	if err != nil || got == nil || got.Session == nil {
		created, err := sessions.Create(ctx, &session.CreateRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, State: state,
		})
		if err != nil {
			return false, fmt.Errorf("agentdispatch: create conversation session %q for %q: %w",
				sessionID, req.ExecutionID, err)
		}
		if created == nil || created.Session == nil {
			return false, fmt.Errorf("agentdispatch: create conversation session %q for %q returned no session",
				sessionID, req.ExecutionID)
		}
		sess = created.Session
	} else {
		sess = got.Session
		if hasDispatchTurn(sess.Events(), invocationID) {
			return true, nil
		}
	}
	turn := session.NewEvent(invocationID)
	turn.Author = "user"
	turn.Content = &genai.Content{Role: "user", Parts: []*genai.Part{{Text: text}}}
	turn.Actions.StateDelta = state
	if err := sessions.AppendEvent(ctx, sess, turn); err != nil {
		return false, fmt.Errorf("agentdispatch: append turn %q to conversation %q: %w",
			req.ExecutionID, sessionID, err)
	}
	return false, nil
}

// hasDispatchTurn reports whether the conversation already records the user
// turn of the given dispatch invocation (scanned from the newest event, where
// a redelivered dispatch's turn is).
func hasDispatchTurn(events session.Events, invocationID string) bool {
	if events == nil || invocationID == "" {
		return false
	}
	for i := events.Len() - 1; i >= 0; i-- {
		ev := events.At(i)
		if ev != nil && ev.Author == "user" && ev.InvocationID == invocationID {
			return true
		}
	}
	return false
}

// sessionIDFor keys the session on the dispatch, so a redelivery reuses a name
// rather than accumulating sessions in the in-memory service.
func sessionIDFor(req Request) string {
	if id := strings.TrimSpace(req.ExecutionID); id != "" {
		return "dispatch:" + id
	}
	return "dispatch:" + req.IdempotencyKey
}

// userIDFor mirrors the HTTP executor's tenant-derived user id.
func userIDFor(req Request) string {
	gcid := strings.TrimSpace(req.GCID)
	if gcid == "" {
		gcid = "anon"
	}
	return req.TenantID + ":" + gcid
}
