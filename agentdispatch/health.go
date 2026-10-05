package agentdispatch

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
)

// EnvHealthPort names the TCP port the health server listens on. Unset means
// 8080, the port every agent Deployment already probes.
const EnvHealthPort = "AGENT_HEALTH_PORT"

const defaultHealthPort = 8080

// ResolveHealthAddr returns the listen address ":<port>" for the health
// server from AGENT_HEALTH_PORT. Anything that is not a port in 1..65535 is an
// error rather than a fallback: a typo here would otherwise put the probes on a
// port nothing listens on and kubelet would kill a working subscriber.
func ResolveHealthAddr() (string, error) {
	raw := strings.TrimSpace(os.Getenv(EnvHealthPort))
	if raw == "" {
		return fmt.Sprintf(":%d", defaultHealthPort), nil
	}
	port, err := strconv.Atoi(raw)
	if err != nil || port < 1 || port > 65535 {
		return "", fmt.Errorf("agentdispatch: %s=%q is not a TCP port in 1..65535", EnvHealthPort, raw)
	}
	return fmt.Sprintf(":%d", port), nil
}

// Health is the readiness flag behind /readyz and the handler that serves the
// two probe paths. It is the ONLY HTTP surface of a subscriber-only agent
// (ADR-254 D6): /healthz says the process is up, /readyz says the dispatch
// subscriber is receiving, and every other path is a 404, which is what makes
// the removal of the ADK web launcher observable from outside the pod.
type Health struct {
	ready atomic.Bool
}

// NewHealth returns a Health that is not ready.
func NewHealth() *Health { return &Health{} }

// SetReady flips /readyz between 200 (true) and 503 (false).
func (h *Health) SetReady(v bool) { h.ready.Store(v) }

// Ready reports the current readiness flag.
func (h *Health) Ready() bool { return h.ready.Load() }

// Handler serves GET/HEAD /healthz (always 200) and GET/HEAD /readyz (200
// while ready, 503 otherwise). Other methods are 405, other paths 404.
func (h *Health) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "ok\n")
	})
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		if h.Ready() {
			w.WriteHeader(http.StatusOK)
			_, _ = io.WriteString(w, "ready\n")
			return
		}
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, "not ready: dispatch subscriber is not receiving\n")
	})
	return mux
}
