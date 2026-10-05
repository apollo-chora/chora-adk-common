package agentdispatch

import (
	"context"
	"errors"
	"testing"

	"google.golang.org/adk/session"
)

// A session service that behaves like ADK's in-memory one: creating a session
// id that already exists is an error.
type dupSessions struct {
	created []string
	deleted []string
}

func (s *dupSessions) Create(_ context.Context, r *session.CreateRequest) (*session.CreateResponse, error) {
	for _, id := range s.created {
		if id == r.SessionID {
			return nil, errors.New("session " + r.SessionID + " already exists")
		}
	}
	s.created = append(s.created, r.SessionID)
	return &session.CreateResponse{}, nil
}
func (s *dupSessions) Delete(_ context.Context, r *session.DeleteRequest) error {
	s.deleted = append(s.deleted, r.SessionID)
	kept := s.created[:0]
	for _, id := range s.created {
		if id != r.SessionID {
			kept = append(kept, id)
		}
	}
	s.created = kept
	return nil
}
func (s *dupSessions) Get(context.Context, *session.GetRequest) (*session.GetResponse, error) {
	return nil, errors.New("not used")
}
func (s *dupSessions) List(context.Context, *session.ListRequest) (*session.ListResponse, error) {
	return nil, errors.New("not used")
}
func (s *dupSessions) AppendEvent(context.Context, session.Session, *session.Event) error {
	return nil
}

// A REDELIVERED dispatch must be runnable. The event bus is at-least-once,
// and the handler retries a transient agent failure by NACKing, so the second
// delivery arrives with the same execution id and therefore the same session
// id. Seen live on 2026-08-20: attempt 1 failed, and every retry then died on
// "session dispatch:<id> already exists" instead of on the original cause —
// so a lane that hit one transient error could never recover, and the message
// dead-lettered five attempts later reporting the wrong reason.
func TestASessionIsReusableAcrossRedelivery(t *testing.T) {
	svc := &dupSessions{}
	req := Request{
		AgentRole: "oe_evaluate", ExecutionID: "sub-1:tsq-1:1",
		TenantID: "11111111-1111-7111-8111-111111111111", GCID: "g1",
		InputPayload: `{"mode":"evaluate"}`,
	}

	if err := prepareSession(context.Background(), svc, "app", req); err != nil {
		t.Fatalf("first delivery: %v", err)
	}
	if err := prepareSession(context.Background(), svc, "app", req); err != nil {
		t.Fatalf("redelivery: %v — a retry can never succeed", err)
	}
	if len(svc.deleted) == 0 {
		t.Error("the stale session was not cleared before re-creating")
	}
}

func TestAMissingTenantStillFailsBeforeTouchingTheSessionService(t *testing.T) {
	svc := &dupSessions{}
	req := Request{AgentRole: "oe_evaluate", ExecutionID: "e1", InputPayload: "{}"}
	if err := prepareSession(context.Background(), svc, "app", req); err == nil {
		t.Fatal("accepted a request with no tenant_id")
	}
	if len(svc.created) != 0 {
		t.Error("created a session for a request that should have been refused")
	}
}
