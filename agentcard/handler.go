package agentcard

import (
	"encoding/json"
	"net/http"
)

// DefaultCacheControl is the Cache-Control header value set by Handler.
// Short TTL because agent capabilities CAN change at deploy time —
// stale Agent Cards lead to A2A integration failures.
const DefaultCacheControl = "public, max-age=60"

// Handler serves the Card on `GET /.well-known/agent-card` per A2A v1.0.
// Other methods get 405 Method Not Allowed with an Allow: GET header.
//
// Production crew launchers wire this onto their public HTTP surface
// (typically a2a.chora.site per ingress topology). The handler is
// stateless — safe to use a single instance under heavy concurrency.
type Handler struct {
	body         []byte
	cacheControl string
}

// NewHandler precomputes the JSON body once at construction. Returns
// nil if the card cannot be marshalled (which only happens with
// non-serialisable fields — should not be possible with the canonical
// Card struct).
func NewHandler(card Card) *Handler {
	body, err := json.Marshal(card)
	if err != nil {
		// Card has no non-serialisable fields, so this is unreachable
		// under the canonical struct. Empty body is the safest fallback
		// — the handler will return an empty JSON object on GET, which
		// is loudly visible at the A2A gateway side.
		body = []byte(`{}`)
	}
	return &Handler{
		body:         body,
		cacheControl: DefaultCacheControl,
	}
}

// WithCacheControl returns a copy of the handler with a custom
// Cache-Control header value.
func (h *Handler) WithCacheControl(value string) *Handler {
	cp := *h
	cp.cacheControl = value
	return &cp
}

// ServeHTTP implements http.Handler. Returns 200 with JSON body on GET,
// 405 on every other method.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		http.Error(w, "agent-card: GET only", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	if h.cacheControl != "" {
		w.Header().Set("Cache-Control", h.cacheControl)
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(h.body)
}

// RegisterMux registers a Handler for `card` on `/.well-known/agent-card`
// of the provided ServeMux. Convenience for crew launchers — one call
// instead of new+register.
func RegisterMux(mux *http.ServeMux, card Card) {
	mux.Handle(A2AWellKnownPath, NewHandler(card))
}
