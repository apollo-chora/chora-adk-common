package agentdispatch

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// ADR-254 D5/D6: transient vs permanent is classified. A permanent run
// failure (unknown discriminator, a gateway FAILED_PRECONDITION such as
// surface_unstamped or companion_suspended) is reported on the wire at once,
// not after four pointless redeliveries.

func TestPermanentError_wrapsAndUnwraps(t *testing.T) {
	base := errors.New("boom")
	err := Permanent("unknown_task_kind: bogus", base)
	var perm *PermanentError
	if !errors.As(err, &perm) || perm.Reason != "unknown_task_kind: bogus" {
		t.Fatalf("Permanent did not produce an errors.As-able PermanentError: %v", err)
	}
	if !errors.Is(err, base) {
		t.Errorf("PermanentError must unwrap to its cause")
	}
	if !strings.Contains(err.Error(), "unknown_task_kind: bogus") {
		t.Errorf("Error() = %q", err)
	}
	if Permanent("", nil) == nil {
		t.Errorf("Permanent with no cause must still be a non-nil permanent error")
	}
}

func TestAPermanentRunErrorPublishesFAILEDImmediately(t *testing.T) {
	pub := &fakePublisher{}
	msg := req(t) // attempt 1 of 5
	h := handler(func(context.Context, Request) (string, error) {
		return "", Permanent("unknown_task_kind: bogus", nil)
	}, pub)

	h.Handle(context.Background(), msg)

	if len(pub.published) != 1 || pub.published[0].body.Status != StatusFailed {
		t.Fatalf("want one FAILED completion on attempt 1, got %+v", pub.published)
	}
	if got := pub.published[0].body.ErrorMessage; !strings.Contains(got, "unknown_task_kind: bogus") {
		t.Errorf("error_message = %q, want the permanent reason", got)
	}
	if !msg.acked || msg.nacked {
		t.Errorf("acked=%v nacked=%v, want acked (no redelivery of a permanent failure)", msg.acked, msg.nacked)
	}
}

func TestClassifyRunError_gatewayPreconditionIsPermanent(t *testing.T) {
	cases := []struct {
		name string
		err  error
		perm bool
	}{
		{"surface_unstamped", status.Error(codes.FailedPrecondition, "surface_unstamped: stamp the crew id"), true},
		{"companion_suspended", status.Error(codes.FailedPrecondition, "companion_suspended: scope=tenant"), true},
		{"invalid_argument", status.Error(codes.InvalidArgument, "bad request"), true},
		{"suspension_unreadable", status.Error(codes.Unavailable, "suspension_unreadable: store"), false},
		{"deadline", status.Error(codes.DeadlineExceeded, "slow"), false},
		{"plain", errors.New("model gateway 503"), false},
		{"wrapped precondition", fmt.Errorf("agentdispatch: run x: %w", status.Error(codes.FailedPrecondition, "surface_unstamped: x")), true},
		{"already permanent", Permanent("unknown_task_kind: x", nil), true},
	}
	for _, tc := range cases {
		got := classifyRunError(tc.err)
		var perm *PermanentError
		isPerm := errors.As(got, &perm)
		if isPerm != tc.perm {
			t.Errorf("%s: permanent=%v, want %v (err=%v)", tc.name, isPerm, tc.perm, got)
		}
		if tc.perm && !strings.Contains(got.Error(), strings.SplitN(status.Convert(tc.err).Message(), ":", 2)[0]) && tc.name != "already permanent" && tc.name != "invalid_argument" {
			t.Errorf("%s: the permanent reason must carry the gateway token, got %q", tc.name, got)
		}
	}
}
