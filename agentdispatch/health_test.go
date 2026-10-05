package agentdispatch

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// The health port is the ONLY HTTP surface a subscriber-only agent has
// (ADR-254 D6). It answers /healthz and /readyz and nothing else, so the
// absence of the ADK web launcher is observable from outside the pod.

func getStatus(t *testing.T, url string) int {
	t.Helper()
	resp, err := http.Get(url) //nolint:gosec // test-local httptest URL
	if err != nil {
		t.Fatalf("GET %s: %v", url, err)
	}
	defer resp.Body.Close()
	return resp.StatusCode
}

func TestHealth_healthzIsAlwaysOK(t *testing.T) {
	h := NewHealth()
	srv := httptest.NewServer(h.Handler())
	defer srv.Close()

	for _, ready := range []bool{false, true, false} {
		h.SetReady(ready)
		if got := getStatus(t, srv.URL+"/healthz"); got != http.StatusOK {
			t.Errorf("ready=%v: /healthz = %d, want 200", ready, got)
		}
	}
}

func TestHealth_readyzFollowsTheSubscriber(t *testing.T) {
	h := NewHealth()
	srv := httptest.NewServer(h.Handler())
	defer srv.Close()

	if got := getStatus(t, srv.URL+"/readyz"); got != http.StatusServiceUnavailable {
		t.Errorf("before the subscriber receives: /readyz = %d, want 503", got)
	}
	h.SetReady(true)
	if got := getStatus(t, srv.URL+"/readyz"); got != http.StatusOK {
		t.Errorf("while the subscriber receives: /readyz = %d, want 200", got)
	}
	h.SetReady(false)
	if got := getStatus(t, srv.URL+"/readyz"); got != http.StatusServiceUnavailable {
		t.Errorf("after the subscriber stopped: /readyz = %d, want 503", got)
	}
	if h.Ready() {
		t.Errorf("Ready() = true after SetReady(false)")
	}
}

func TestHealth_servesNothingElse(t *testing.T) {
	// D6: no agent serves HTTP. Every path the ADK web launcher used to answer
	// must be a 404 on the health port, which is what makes the launcher's
	// removal checkable from a shell.
	h := NewHealth()
	srv := httptest.NewServer(h.Handler())
	defer srv.Close()

	for _, p := range []string{"/", "/api/reasoning_engine", "/api/stream_reasoning_engine", "/run", "/run_sse", "/list-apps", "/healthz/", "/readyz/x"} {
		if got := getStatus(t, srv.URL+p); got != http.StatusNotFound {
			t.Errorf("GET %s = %d, want 404", p, got)
		}
	}
}

func TestHealth_probesAreReadOnly(t *testing.T) {
	h := NewHealth()
	srv := httptest.NewServer(h.Handler())
	defer srv.Close()

	for _, p := range []string{"/healthz", "/readyz"} {
		resp, err := http.Post(srv.URL+p, "text/plain", nil) //nolint:gosec // test-local
		if err != nil {
			t.Fatalf("POST %s: %v", p, err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusMethodNotAllowed {
			t.Errorf("POST %s = %d, want 405", p, resp.StatusCode)
		}
	}
}

func TestResolveHealthAddr(t *testing.T) {
	t.Setenv(EnvHealthPort, "")
	if got, err := ResolveHealthAddr(); err != nil || got != ":8080" {
		t.Errorf("unset %s: got %q, %v; want \":8080\", nil", EnvHealthPort, got, err)
	}
	t.Setenv(EnvHealthPort, "9090")
	if got, err := ResolveHealthAddr(); err != nil || got != ":9090" {
		t.Errorf("%s=9090: got %q, %v; want \":9090\", nil", EnvHealthPort, got, err)
	}
	for _, bad := range []string{"abc", "0", "70000", "-1", ":8080"} {
		t.Setenv(EnvHealthPort, bad)
		if _, err := ResolveHealthAddr(); err == nil {
			t.Errorf("%s=%q: want an error, got nil", EnvHealthPort, bad)
		}
	}
}
