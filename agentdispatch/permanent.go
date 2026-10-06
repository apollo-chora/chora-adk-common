package agentdispatch

import (
	"errors"
	"fmt"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// PermanentError marks a run failure that no redelivery can fix (ADR-254 D5/D6:
// transient vs permanent is classified). The handler reports it on the wire as
// a FAILED completion at once and acks the request, instead of NACKing it four
// more times through the backoff ladder while the parked run hears nothing.
//
// Agents return it for a permanent fault they can name (an unknown payload
// discriminator, a missing required input), and classifyRunError derives it from
// the gateway's FAILED_PRECONDITION answers (surface_unstamped,
// companion_suspended) and INVALID_ARGUMENT; UNAVAILABLE (for instance
// suspension_unreadable) and every other code stay retryable.
type PermanentError struct {
	// Reason is what the completion's error_message carries: a stable token,
	// optionally followed by ": detail".
	Reason string
	Err    error
}

func (e *PermanentError) Error() string {
	switch {
	case e.Reason != "" && e.Err != nil:
		return e.Reason + ": " + e.Err.Error()
	case e.Reason != "":
		return e.Reason
	case e.Err != nil:
		return e.Err.Error()
	}
	return "permanent failure"
}

// Unwrap exposes the cause to errors.Is / errors.As.
func (e *PermanentError) Unwrap() error { return e.Err }

// Permanent wraps a cause as a PermanentError carrying reason as the wire
// message. A nil cause is allowed: the reason alone is the failure.
func Permanent(reason string, cause error) error {
	return &PermanentError{Reason: reason, Err: cause}
}

// classifyRunError turns a run error into a PermanentError when the failure
// cannot succeed on redelivery, and returns it unchanged otherwise. The gRPC
// codes are the model gateway's contract (ADR-254 D7 / WP-D wire note):
// FailedPrecondition carries surface_unstamped / companion_suspended,
// InvalidArgument a malformed request; both are permanent. Unavailable
// (suspension_unreadable) and the rest are transient.
func classifyRunError(err error) error {
	if err == nil {
		return nil
	}
	var perm *PermanentError
	if errors.As(err, &perm) {
		return err
	}
	switch status.Code(err) {
	case codes.FailedPrecondition, codes.InvalidArgument:
		return &PermanentError{Reason: status.Convert(err).Message(), Err: err}
	}
	return err
}

// isPermanent reports whether err is (or wraps) a PermanentError.
func isPermanent(err error) bool {
	var perm *PermanentError
	return errors.As(err, &perm)
}

// permanentReason is the wire message for a permanent error.
func permanentReason(err error) string {
	var perm *PermanentError
	if errors.As(err, &perm) {
		return perm.Error()
	}
	return fmt.Sprintf("permanent: %v", err)
}
