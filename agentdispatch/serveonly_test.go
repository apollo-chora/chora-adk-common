package agentdispatch

import (
	"context"
	"errors"
	"net"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// RunSubscriberOnly is the whole boot of a subscriber-only agent (ADR-254 D6):
// refuse to start without the one transport that exists, serve the health
// port, flip readiness around the receive loop, and return the subscriber's
// error so main() exits non-zero and the pod dies rather than idling healthy.

func onlyServeConfig() ServeConfig {
	return ServeConfig{
		AgentRole:   "oe_evaluate",
		ServiceName: "chora-oe-evaluator",
		AppName:     "test-app",
		RootAgent:   fakeRootAgent{},
		Sessions:    session.InMemoryService(),
	}
}

// fakeRootAgent satisfies agent.Agent for wiring tests; it is never run.
type fakeRootAgent struct{ agent.Agent }

func (fakeRootAgent) Name() string { return "fake_root" }

func localListener(t *testing.T) net.Listener {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	return ln
}

func TestRunSubscriberOnly_refusesWhenDispatchIsDisabled(t *testing.T) {
	for _, v := range []string{"", "false", "0", "no"} {
		t.Setenv(EnvDispatchEnabled, v)
		t.Setenv(EnvNATSURL, "nats://nats:4222")
		called := false
		err := RunSubscriberOnly(context.Background(), onlyServeConfig(), RunOptions{
			listener: localListener(t),
			serve:    func(context.Context, ServeConfig) error { called = true; return nil },
		})
		if err == nil {
			t.Fatalf("%s=%q: RunSubscriberOnly returned nil; a subscriber-only binary has no other transport", EnvDispatchEnabled, v)
		}
		if !strings.Contains(err.Error(), EnvDispatchEnabled) {
			t.Errorf("%s=%q: error %q does not name the env var", EnvDispatchEnabled, v, err)
		}
		if called {
			t.Errorf("%s=%q: the subscriber was started despite the refusal", EnvDispatchEnabled, v)
		}
	}
}

func TestRunSubscriberOnly_refusesAnEmptyRole(t *testing.T) {
	t.Setenv(EnvDispatchEnabled, "true")
	t.Setenv(EnvNATSURL, "nats://nats:4222")
	cfg := onlyServeConfig()
	cfg.AgentRole = ""
	err := RunSubscriberOnly(context.Background(), cfg, RunOptions{
		listener: localListener(t),
		serve:    func(context.Context, ServeConfig) error { t.Fatal("serve must not run"); return nil },
	})
	if err == nil || !strings.Contains(err.Error(), "AgentRole") {
		t.Fatalf("want an AgentRole error, got %v", err)
	}
}

func TestRunSubscriberOnly_refusesWithoutABusURL(t *testing.T) {
	t.Setenv(EnvDispatchEnabled, "true")
	t.Setenv(EnvNATSURL, "")
	err := RunSubscriberOnly(context.Background(), onlyServeConfig(), RunOptions{
		listener: localListener(t),
		serve:    func(context.Context, ServeConfig) error { t.Fatal("serve must not run"); return nil },
	})
	if err == nil || !strings.Contains(err.Error(), EnvNATSURL) {
		t.Fatalf("want a %s error, got %v", EnvNATSURL, err)
	}
}

func TestRunSubscriberOnly_healthBindFailureIsFatal(t *testing.T) {
	t.Setenv(EnvDispatchEnabled, "true")
	t.Setenv(EnvNATSURL, "nats://nats:4222")
	taken := localListener(t)
	defer taken.Close()
	err := RunSubscriberOnly(context.Background(), onlyServeConfig(), RunOptions{
		HealthAddr: taken.Addr().String(),
		serve: func(context.Context, ServeConfig) error {
			t.Fatal("serve must not run when the health port cannot bind")
			return nil
		},
	})
	if err == nil || !strings.Contains(err.Error(), taken.Addr().String()) {
		t.Fatalf("want a bind error naming %s, got %v", taken.Addr(), err)
	}
}

func TestRunSubscriberOnly_readinessFollowsTheReceiveLoop(t *testing.T) {
	t.Setenv(EnvDispatchEnabled, "true")
	t.Setenv(EnvNATSURL, "nats://nats:4222")
	ln := localListener(t)
	base := "http://" + ln.Addr().String()
	errBoom := errors.New("boom: streaming pull refused")

	var sawReadyDuringServe atomic.Bool
	err := RunSubscriberOnly(context.Background(), onlyServeConfig(), RunOptions{
		listener: ln,
		serve: func(_ context.Context, cfg ServeConfig) error {
			// RunSubscriberOnly flips ready right before handing over to Serve
			// and back right after it returns; the receive loop IS the serve.
			if getStatus(t, base+"/healthz") != http.StatusOK {
				t.Errorf("/healthz != 200 while serving")
			}
			sawReadyDuringServe.Store(getStatus(t, base+"/readyz") == http.StatusOK)
			return errBoom
		},
	})
	if !sawReadyDuringServe.Load() {
		t.Errorf("/readyz was not 200 while the receive loop ran")
	}
	if !errors.Is(err, errBoom) {
		t.Fatalf("the subscriber's error must surface to main (pod death); got %v", err)
	}
	if err == nil || !strings.Contains(err.Error(), "oe_evaluate") {
		t.Errorf("error %v does not name the role", err)
	}
	// The health server is gone with the process: a probe now fails to connect.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := http.Get(base + "/healthz"); err != nil { //nolint:gosec // test-local
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Errorf("health server still answering after RunSubscriberOnly returned")
}

func TestRunSubscriberOnly_cleanShutdownReturnsNil(t *testing.T) {
	t.Setenv(EnvDispatchEnabled, "true")
	t.Setenv(EnvNATSURL, "nats://nats:4222")
	ctx, cancel := context.WithCancel(context.Background())
	err := RunSubscriberOnly(ctx, onlyServeConfig(), RunOptions{
		listener: localListener(t),
		serve: func(ctx context.Context, cfg ServeConfig) error {
			cancel()
			<-ctx.Done()
			return nil // Receive returns nil on a cancelled context
		},
	})
	if err != nil {
		t.Fatalf("clean shutdown must return nil, got %v", err)
	}
}

func TestRunSubscriberOnly_firstPullFailuresAreNamedInTheExitLine(t *testing.T) {
	// No consumer preflight: the first subscribe is the check. Its two
	// operator-fixable failures must reach main() with the subscription and
	// the fix in the message.
	t.Setenv(EnvDispatchEnabled, "true")
	t.Setenv(EnvNATSURL, "nats://nats:4222")
	t.Setenv(EnvDispatchSubscription, "")
	derived := RequestSubscription("chora-oe-evaluator", "oe_evaluate")
	full := "bus nats://nats:4222 subscription " + derived

	err := RunSubscriberOnly(context.Background(), onlyServeConfig(), RunOptions{
		listener: localListener(t),
		serve:    func(context.Context, ServeConfig) error { return notFoundErr() },
	})
	if err == nil || !strings.Contains(err.Error(), full) || !strings.Contains(err.Error(), "does not exist") ||
		!strings.Contains(err.Error(), EnvDispatchSubscription) {
		t.Errorf("missing subscription: want the full name, 'does not exist' and the env var; got %v", err)
	}

	err = RunSubscriberOnly(context.Background(), onlyServeConfig(), RunOptions{
		listener: localListener(t),
		serve:    func(context.Context, ServeConfig) error { return permissionDeniedErr() },
	})
	if err == nil || !strings.Contains(err.Error(), full) || !strings.Contains(err.Error(), "not consumable") {
		t.Errorf("denied subscription: want the full name and the consumability fix; got %v", err)
	}

	// An explicit AGENT_DISPATCH_SUBSCRIPTION is the name the exit line carries.
	t.Setenv(EnvDispatchSubscription, "chora-oe-evaluator-shadow.agent-dispatch-eval-oe-evaluate-requested")
	err = RunSubscriberOnly(context.Background(), onlyServeConfig(), RunOptions{
		listener: localListener(t),
		serve:    func(context.Context, ServeConfig) error { return notFoundErr() },
	})
	if err == nil || !strings.Contains(err.Error(), "chora-oe-evaluator-shadow.agent-dispatch-eval-oe-evaluate-requested") {
		t.Errorf("explicit subscription not named in the exit line: %v", err)
	}
}

func notFoundErr() error         { return status.Error(codes.NotFound, "Resource not found") }
func permissionDeniedErr() error { return status.Error(codes.PermissionDenied, "User not authorized") }
