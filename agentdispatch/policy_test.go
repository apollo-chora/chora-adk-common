package agentdispatch

import (
	"context"
	"testing"

	"google.golang.org/adk/session"
)

// ADR-254 D6 companion_chat: a conversational role keeps ONE session per
// conversation across dispatches; every other role keeps the stateless
// per-dispatch session. Both policies are exercised against the real in-memory
// session service so the contract is the service's, not a fake's.

func chatReq(execID, convID, text string) Request {
	return Request{
		AgentRole: "companion_chat", ExecutionID: execID, TenantID: "t1", GCID: "g1",
		ThreadID: convID, IdempotencyKey: "agent_dispatch.companion_chat." + execID,
		InputPayload: `{"conversation_id":"` + convID + `","message":"` + text + `"}`,
	}
}

func TestStatelessPolicyIsTheDefault(t *testing.T) {
	userID, sessionID, persistent := resolveSessionKey(RunPolicy{}, chatReq("e1", "c1", "hi"))
	if persistent || sessionID != "dispatch:e1" || userID != "t1:g1" {
		t.Errorf("default policy: user=%q session=%q persistent=%v", userID, sessionID, persistent)
	}
	// A SessionKey that declines (ok=false) also falls back to stateless.
	_, sessionID, persistent = resolveSessionKey(RunPolicy{SessionKey: func(Request) (string, string, bool) { return "", "", false }}, chatReq("e1", "c1", "hi"))
	if persistent || sessionID != "dispatch:e1" {
		t.Errorf("declined key must fall back: session=%q persistent=%v", sessionID, persistent)
	}
}

func TestPersistentPolicyKeepsTheConversationAcrossDispatches(t *testing.T) {
	ctx := context.Background()
	svc := session.InMemoryService()
	policy := RunPolicy{SessionKey: func(r Request) (string, string, bool) { return r.GCID, r.ThreadID, true }}
	const app = "companion_chat"

	// Turn 1 creates the conversation and records the learner's turn ONCE,
	// under the dispatch's invocation id, with the payload as its state delta.
	r1 := chatReq("turn-1", "conv-7", "hello")
	u, sid, persistent := resolveSessionKey(policy, r1)
	if !persistent || u != "g1" || sid != "conv-7" {
		t.Fatalf("persistent key: user=%q session=%q persistent=%v", u, sid, persistent)
	}
	resumed, err := prepareConversationTurn(ctx, svc, app, r1, u, sid, "hello")
	if err != nil || resumed {
		t.Fatalf("turn 1: resumed=%v err=%v", resumed, err)
	}
	got, err := svc.Get(ctx, &session.GetRequest{AppName: app, UserID: u, SessionID: sid})
	if err != nil || got == nil || got.Session == nil {
		t.Fatalf("conversation not created: %v", err)
	}
	if n := got.Session.Events().Len(); n != 1 {
		t.Fatalf("turn 1 must record exactly one user turn, got %d events", n)
	}
	first := got.Session.Events().At(0)
	if first.Author != "user" || first.InvocationID != "dispatch:turn-1" || first.Content == nil || first.Content.Parts[0].Text != "hello" {
		t.Fatalf("turn 1 event wrong: author=%q invocation=%q content=%v", first.Author, first.InvocationID, first.Content)
	}

	// A REDELIVERY of turn 1 (same execution id) must not append the learner's
	// message again: it resumes on the history as it stands.
	resumed, err = prepareConversationTurn(ctx, svc, app, r1, u, sid, "hello")
	if err != nil || !resumed {
		t.Fatalf("redelivered turn 1: resumed=%v err=%v", resumed, err)
	}
	got, _ = svc.Get(ctx, &session.GetRequest{AppName: app, UserID: u, SessionID: sid})
	if n := got.Session.Events().Len(); n != 1 {
		t.Fatalf("a redelivery appended the turn again: %d events", n)
	}

	// Simulate the dialogue the first run produced.
	ev := session.NewEvent("inv-1")
	ev.Author = "companion_chat"
	if err := svc.AppendEvent(ctx, got.Session, ev); err != nil {
		t.Fatal(err)
	}

	// Turn 2 on the SAME conversation keeps the history, appends its own turn
	// and merges the new payload as state.
	r2 := chatReq("turn-2", "conv-7", "and then?")
	resumed, err = prepareConversationTurn(ctx, svc, app, r2, u, sid, "and then?")
	if err != nil || resumed {
		t.Fatalf("turn 2: resumed=%v err=%v", resumed, err)
	}
	got, err = svc.Get(ctx, &session.GetRequest{AppName: app, UserID: u, SessionID: sid})
	if err != nil || got == nil || got.Session == nil {
		t.Fatalf("conversation lost on the second dispatch: %v", err)
	}
	if n := got.Session.Events().Len(); n != 3 {
		t.Errorf("turn 2 must keep the earlier dialogue and add one turn, got %d events", n)
	}
	last := got.Session.Events().At(got.Session.Events().Len() - 1)
	if last.Author != "user" || last.InvocationID != "dispatch:turn-2" {
		t.Errorf("turn 2 event wrong: author=%q invocation=%q", last.Author, last.InvocationID)
	}
	if msg, _ := got.Session.State().Get("message"); msg != "and then?" {
		t.Errorf("turn 2 state not merged: message=%v", msg)
	}
	if key, _ := got.Session.State().Get("dispatch_idempotency_key"); key != r2.IdempotencyKey {
		t.Errorf("dispatch key must follow the latest dispatch, got %v", key)
	}
	// And turn 2's redelivery resumes too.
	if resumed, err = prepareConversationTurn(ctx, svc, app, r2, u, sid, "and then?"); err != nil || !resumed {
		t.Fatalf("redelivered turn 2: resumed=%v err=%v", resumed, err)
	}

	// The stateless policy on another role still clears and recreates.
	r3 := chatReq("turn-3", "conv-7", "x")
	if err := prepareStatelessSession(ctx, svc, app, r3, userIDFor(r3), sessionIDFor(r3)); err != nil {
		t.Fatal(err)
	}
	if err := prepareStatelessSession(ctx, svc, app, r3, userIDFor(r3), sessionIDFor(r3)); err != nil {
		t.Fatalf("stateless redelivery must recreate cleanly: %v", err)
	}
}

func TestHasDispatchTurn(t *testing.T) {
	ctx := context.Background()
	svc := session.InMemoryService()
	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "a", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}
	if hasDispatchTurn(created.Session.Events(), "dispatch:x") || hasDispatchTurn(nil, "dispatch:x") {
		t.Fatal("empty history cannot hold a turn")
	}
	agentEv := session.NewEvent("dispatch:x")
	agentEv.Author = "companion_chat"
	if err := svc.AppendEvent(ctx, created.Session, agentEv); err != nil {
		t.Fatal(err)
	}
	got, _ := svc.Get(ctx, &session.GetRequest{AppName: "a", UserID: "u", SessionID: "s"})
	if hasDispatchTurn(got.Session.Events(), "dispatch:x") {
		t.Fatal("only a USER event counts as the dispatch's turn")
	}
	if hasDispatchTurn(got.Session.Events(), "") {
		t.Fatal("an empty invocation id never matches")
	}
}

func TestTerminalAuthorKnowsTheLaneRolesAndTheHTTPEraNames(t *testing.T) {
	for role, want := range map[string]string{
		"oe_evaluate": "oe_evaluator", "oe_moderate": "oe_moderator",
		"qgen_question": "qgen_question_generation", "qgen_generate": "",
		"qgen_critic": "qgen_critic", "qgen_critique": "qgen_critic",
		"qgen_render": "", "companion_chat": "", "kg_explore": "", "companion_diagnose": "",
	} {
		if got := TerminalAuthor(role); got != want {
			t.Errorf("TerminalAuthor(%q) = %q, want %q", role, got, want)
		}
	}
}
