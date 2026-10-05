// Package tenancy is the shared adapter for the TenantManaPool aggregate
// (per ADR-142). ALL ADK Go crews import this — Familiar, Question
// Generation, and any future crew per Tier 5 D20 crew-composition.
//
// Production wiring will be a gRPC client to
// chora-tenancy.ManaPoolService. POC stub returns deterministic data
// (last hex digit of GCID drives balance; digit 0 -> mana 0).
//
// Per feedback_no_inline_config: endpoint URL from env, never inline.
package tenancy

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// ManaBalance is what we read pre-call to decide whether to allow the
// upstream model call.
type ManaBalance struct {
	UserGCID      string
	RemainingMana int
	Tier          string // "basic" | "standard" | "premium"
}

// DebitRequest captures the post-call accounting we send to the
// TenantManaPool (idempotent on session+agent+user).
type DebitRequest struct {
	UserGCID  string
	Tokens    int
	AgentKind string // crew + agent identifier; e.g., "familiar.companion" or "qgen.validator"
	CrewKind  string // optional crew discriminator; e.g., "familiar" | "qgen"
	SessionID string
}

// ManaClient is implemented by the production gRPC client AND the POC
// stub. Hexagonal port boundary — used by every crew's manaplugin.
type ManaClient interface {
	PeekBalance(ctx context.Context, userGCID string) (ManaBalance, error)
	Debit(ctx context.Context, req DebitRequest) error
}

// New returns a ManaClient based on the TENANCY_GRPC_ENDPOINT env var.
//
//   - "stub://..." -> in-memory stub (POC + tests)
//   - other       -> gRPC client (production; not implemented yet)
//
// Per feedback_no_inline_config: this is the ONLY place that touches
// the endpoint env var.
func New() (ManaClient, error) {
	endpoint := os.Getenv("TENANCY_GRPC_ENDPOINT")
	if endpoint == "" {
		return nil, errors.New(
			"TENANCY_GRPC_ENDPOINT not set; refusing inline default " +
				"per feedback_no_inline_config")
	}
	if strings.HasPrefix(endpoint, "stub://") {
		return &stubManaClient{}, nil
	}
	return nil, fmt.Errorf(
		"production gRPC ManaClient not implemented yet — "+
			"wire to chora-tenancy.ManaPoolService at sandbox time (endpoint=%q)",
		endpoint)
}

// stubManaClient — deterministic for tests. Balance derived from the
// last hex digit of the GCID so tests can simulate insufficient-balance
// users (digit 0 -> mana 0).
type stubManaClient struct{}

func (s *stubManaClient) PeekBalance(_ context.Context, userGCID string) (ManaBalance, error) {
	if userGCID == "" {
		return ManaBalance{}, errors.New("userGCID required")
	}
	remaining := 1000
	if len(userGCID) > 0 {
		last := userGCID[len(userGCID)-1:]
		if v, err := strconv.ParseInt(last, 16, 32); err == nil {
			if v == 0 {
				remaining = 0
			} else {
				remaining = 100 + int(v)*1000
			}
		}
	}
	return ManaBalance{UserGCID: userGCID, RemainingMana: remaining, Tier: "basic"}, nil
}

func (s *stubManaClient) Debit(_ context.Context, req DebitRequest) error {
	if req.UserGCID == "" {
		return errors.New("userGCID required")
	}
	if req.Tokens < 0 {
		return errors.New("tokens must be >= 0")
	}
	return nil
}
