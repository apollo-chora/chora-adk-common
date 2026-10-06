package agentdispatch

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// RunOptions tunes RunSubscriberOnly. The zero value is production: the health
// server binds AGENT_HEALTH_PORT (default 8080) and the real Serve runs. The
// unexported fields are test seams; nothing outside this package can set them,
// so a production binary cannot accidentally run with a fake subscriber.
type RunOptions struct {
	// HealthAddr overrides the listen address resolved from AGENT_HEALTH_PORT.
	HealthAddr string
	Logger     *slog.Logger

	listener net.Listener
	serve    func(ctx context.Context, cfg ServeConfig) error
}

// healthShutdownGrace bounds the health server's drain on exit.
const healthShutdownGrace = 5 * time.Second

// RunSubscriberOnly is the whole boot of a subscriber-only agent (ADR-254 D6).
//
// It refuses to start unless AGENT_DISPATCH_ENABLED parses true, because after
// D6 the dispatch subscriber is the binary's only transport: a pod that "came
// up" without it would be a healthy-looking pod that never consumes. It then
// resolves the subscription (AGENT_DISPATCH_SUBSCRIPTION or the derived name),
// binds the health port, flips /readyz to 200 for exactly as long as Serve
// runs, and returns Serve's error so main() exits non-zero and the pod dies.
// A cancelled context (SIGTERM) is a clean shutdown and returns nil.
//
// There is deliberately NO consumer preflight: the first subscribe is the
// check instead. A failure to create the durable consumer is not retried by
// the bus, Serve returns it within a second, and the error that reaches
// main() is classified below so the exit line still names the subscription
// and the fix.
func RunSubscriberOnly(ctx context.Context, cfg ServeConfig, opts RunOptions) error {
	log := opts.Logger
	if log == nil {
		log = cfg.Logger
	}
	if log == nil {
		log = slog.Default()
	}
	if !Enabled() {
		return fmt.Errorf("agentdispatch: refusing to start: %s must be true; this binary "+
			"is subscriber-only (ADR-254 D6) and has no other transport", EnvDispatchEnabled)
	}
	if strings.TrimSpace(cfg.AgentRole) == "" {
		return errors.New("agentdispatch: RunSubscriberOnly: ServeConfig.AgentRole is required")
	}
	url := strings.TrimSpace(os.Getenv(EnvNATSURL))
	if url == "" {
		return fmt.Errorf("agentdispatch: RunSubscriberOnly: %s is unset", EnvNATSURL)
	}
	subscription := strings.TrimSpace(os.Getenv(EnvDispatchSubscription))
	if subscription == "" {
		subscription = RequestSubscription(cfg.ServiceName, cfg.AgentRole)
	}

	ln := opts.listener
	if ln == nil {
		addr := opts.HealthAddr
		if addr == "" {
			resolved, err := ResolveHealthAddr()
			if err != nil {
				return err
			}
			addr = resolved
		}
		var err error
		ln, err = net.Listen("tcp", addr)
		if err != nil {
			return fmt.Errorf("agentdispatch: health port %s: %w", addr, err)
		}
	}

	health := NewHealth()
	srv := &http.Server{
		Handler:           health.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}
	healthErr := make(chan error, 1)
	go func() { healthErr <- srv.Serve(ln) }()
	defer func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), healthShutdownGrace)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()

	serve := opts.serve
	if serve == nil {
		serve = Serve
	}
	serveCtx, cancelServe := context.WithCancel(ctx)
	defer cancelServe()
	serveErr := make(chan error, 1)
	go func() {
		log.Info("agentdispatch.subscriber_only_boot",
			"role", cfg.AgentRole, "bus", url, "subscription", subscription,
			"health_addr", ln.Addr().String())
		health.SetReady(true)
		err := serve(serveCtx, cfg)
		health.SetReady(false)
		serveErr <- err
	}()

	select {
	case err := <-serveErr:
		if err != nil {
			return describeSubscriberError(cfg.AgentRole, url, subscription, err)
		}
		// The bus subscribes asynchronously, so a nil return means the receive
		// loop is now running — not that the agent finished its work. Block
		// until the process is asked to stop; returning here would exit the
		// container immediately and restart it in a tight loop.
		<-ctx.Done()
		return nil
	case err := <-healthErr:
		// The probe server died underneath a live subscriber. Stop consuming
		// and exit: kubelet would kill the pod on the failed probes anyway, and
		// this way the cause is in the exit line rather than in a probe log.
		cancelServe()
		<-serveErr
		health.SetReady(false)
		return fmt.Errorf("agentdispatch: health server for role %s stopped: %w", cfg.AgentRole, err)
	}
}

// describeSubscriberError wraps the error that ended the receive loop so the
// exit line names the subscription and, for the two first-subscribe failures
// an operator can actually fix, the fix: a NotFound means the name is wrong
// or the consumer was never provisioned, a PermissionDenied means the
// identity may not consume the subscription.
func describeSubscriberError(role, busURL, subscription string, err error) error {
	full := fmt.Sprintf("bus %s subscription %s", busURL, subscription)
	switch status.Code(err) {
	case codes.NotFound:
		return fmt.Errorf("agentdispatch: subscriber for role %s stopped: subscription %s does not exist "+
			"(check %s and the lane provisioning): %w", role, full, EnvDispatchSubscription, err)
	case codes.PermissionDenied:
		return fmt.Errorf("agentdispatch: subscriber for role %s stopped: subscription %s is not consumable "+
			"by this identity: %w", role, full, err)
	default:
		return fmt.Errorf("agentdispatch: subscriber for role %s stopped (subscription %s): %w", role, full, err)
	}
}
