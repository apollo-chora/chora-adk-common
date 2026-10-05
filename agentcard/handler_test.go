package agentcard

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHandler_servesCardAtWellKnownPath(t *testing.T) {
	card, _ := NewBuilder().
		WithName("familiar_companion").
		WithVersion("1.0.0").
		WithAGID("agid:familiar:abc").
		Build()

	h := NewHandler(card)
	req := httptest.NewRequest(http.MethodGet, A2AWellKnownPath, nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("want 200; got %d (body=%s)", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("want application/json content type; got %q", ct)
	}

	var got Card
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("body should be JSON Card; got err %v", err)
	}
	if got.Name != "familiar_companion" {
		t.Errorf("want familiar_companion name in body; got %q", got.Name)
	}
}

func TestHandler_rejectsNonGET(t *testing.T) {
	card, _ := NewBuilder().
		WithName("x").
		WithVersion("1.0.0").
		WithAGID("agid:x").
		Build()
	h := NewHandler(card)

	for _, method := range []string{http.MethodPost, http.MethodPut, http.MethodDelete} {
		req := httptest.NewRequest(method, A2AWellKnownPath, nil)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusMethodNotAllowed {
			t.Errorf("method %s should return 405; got %d", method, rec.Code)
		}
		if rec.Header().Get("Allow") != http.MethodGet {
			t.Errorf("Allow header should be GET; got %q", rec.Header().Get("Allow"))
		}
	}
}

func TestHandler_setsCacheHeaders(t *testing.T) {
	// A2A spec recommends short cache TTL on the well-known path since
	// agent capabilities CAN change at deploy time. We expose a tunable
	// header but the default must be sensible.
	card, _ := NewBuilder().
		WithName("x").
		WithVersion("1.0.0").
		WithAGID("agid:x").
		Build()
	h := NewHandler(card)

	req := httptest.NewRequest(http.MethodGet, A2AWellKnownPath, nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if cc := rec.Header().Get("Cache-Control"); cc == "" {
		t.Error("want Cache-Control header; got empty")
	}
}

func TestRegisterMux_attachesAtWellKnownPath(t *testing.T) {
	// Convenience helper for crew launchers — registers GET on the
	// canonical path in one call.
	card, _ := NewBuilder().
		WithName("familiar").
		WithVersion("1.0.0").
		WithAGID("agid:x").
		Build()

	mux := http.NewServeMux()
	RegisterMux(mux, card)

	req := httptest.NewRequest(http.MethodGet, A2AWellKnownPath, nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("want 200; got %d", rec.Code)
	}
}
