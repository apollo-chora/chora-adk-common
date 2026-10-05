// Package manaplugin builds the shared ADK Go plugin that implements
// the Chora per-user mana gate + Chora-context label injection
// (Pattern A from adk-tool-calling-loop skill).
//
// EVERY ADK Go crew (Familiar, QGen, and any future crew per Tier 5 D20)
// imports this package and registers the plugin in its launcher Config.
// This is the ONLY Chora-side runtime code on the model call path per
// ADR-146 Model Broker full retirement.
//
//   - Per-product caps     -> platform quota service (external to this lib)
//   - Per-tenant caps      -> tenant budget + event-driven enforcement
//   - Per-user caps        -> THIS PLUGIN (TenantManaPool, ADR-142)
//
// Per-tenant cost attribution: labels (tenant_id, user_gcid, crew_kind,
// agent_kind, chora_env) propagate to the cost-attribution sink for
// IMDA D3 evidence stream.
package manaplugin

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"

	"google.golang.org/genai"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/plugin"

	"github.com/apollo-chora/chora-adk-common/tenancy"
)

// Config configures the plugin for a specific crew.
type Config struct {
	// Mana is the TenantManaPool client (gRPC in prod; stub in POC).
	Mana tenancy.ManaClient

	// CrewKind identifies the crew (e.g., "familiar", "qgen", "delivery_rostering").
	// Persisted into session state + emitted as cost-attribution label.
	// REQUIRED — refuses to build a plugin without it (multi-crew safety).
	CrewKind string

	// DefaultEstimatedTokens is the pre-call mana-gate token estimate
	// when the agent doesn't supply one. Conservative default: 100.
	DefaultEstimatedTokens int
}

// New constructs the mana gate plugin. Returns an error if Config is
// incomplete (missing Mana or CrewKind).
func New(cfg Config) (*plugin.Plugin, error) {
	if cfg.Mana == nil {
		return nil, errors.New("manaplugin: Mana client required")
	}
	if cfg.CrewKind == "" {
		return nil, errors.New(
			"manaplugin: CrewKind required (multi-crew safety — every crew " +
				"must self-identify for cost attribution + IMDA D3 evidence)")
	}
	if cfg.DefaultEstimatedTokens <= 0 {
		cfg.DefaultEstimatedTokens = 100
	}

	return plugin.New(plugin.Config{
		Name: fmt.Sprintf("chora_mana_gate_%s", cfg.CrewKind),
		BeforeRunCallback: func(ic agent.InvocationContext) (*genai.Content, error) {
			state := ic.Session().State()
			tenantID := getStateString(state, "tenant_id")
			userGCID := getStateString(state, "user_gcid")
			agentKind := ic.Agent().Name()

			if tenantID == "" || userGCID == "" {
				return nil, errors.New(
					"session state must include tenant_id + user_gcid " +
						"(IMDA D3 evidence stream + per-tenant cost " +
						"attribution); refusing call")
			}

			balance, err := cfg.Mana.PeekBalance(ic, userGCID)
			if err != nil {
				return nil, fmt.Errorf("mana peek failed for user %s: %w",
					safePrefix(userGCID, 8), err)
			}
			if balance.RemainingMana < cfg.DefaultEstimatedTokens {
				return nil, fmt.Errorf(
					"user %s mana balance %d < requested %d; refusing pre-Vertex-AI call",
					safePrefix(userGCID, 8), balance.RemainingMana,
					cfg.DefaultEstimatedTokens)
			}

			// Persist context for AfterRunCallback debit
			if err := state.Set("chora.crew_kind", cfg.CrewKind); err != nil {
				slog.Warn("failed to set crew_kind", "err", err)
			}
			if err := state.Set("chora.agent_kind", agentKind); err != nil {
				slog.Warn("failed to set agent_kind", "err", err)
			}
			// Persist pre-debit balance for AfterRunCallback span emission.
			// Stored as int — state.Set handles any encoding needed.
			if err := state.Set("chora.mana.balance_pre", balance.RemainingMana); err != nil {
				slog.Warn("failed to set mana balance_pre", "err", err)
			}
			// Per Iter 1 BLANKET (ADR-142 mana tiers + POC W3 Iter 7 cost
			// levers): propagate mana_tier into session state so downstream
			// plugins (tieredmodelplugin) can pick a per-tier Gemini model +
			// MaxOutputTokens cap. Empty Tier acceptable — downstream defaults
			// to "basic" (cheapest, safest under stub conditions).
			if balance.Tier != "" {
				if err := state.Set("mana_tier", balance.Tier); err != nil {
					slog.Warn("failed to set mana_tier", "err", err)
				}
			}
			return nil, nil
		},
		AfterRunCallback: func(ic agent.InvocationContext) {
			state := ic.Session().State()
			userGCID := getStateString(state, "user_gcid")
			agentKind := getStateString(state, "chora.agent_kind")
			crewKind := getStateString(state, "chora.crew_kind")
			if crewKind == "" {
				crewKind = cfg.CrewKind
			}
			if userGCID == "" {
				return
			}

			// POC: debit by default estimate. Sandbox week 2 upgrades
			// this to use the actual token count from the session's
			// recorded events / usage metadata.
			actualTokens := cfg.DefaultEstimatedTokens
			balancePre := getStateInt(state, "chora.mana.balance_pre")
			debitErr := cfg.Mana.Debit(ic, tenancy.DebitRequest{
				UserGCID:  userGCID,
				Tokens:    actualTokens,
				AgentKind: agentKind,
				CrewKind:  crewKind,
			})
			if debitErr != nil {
				slog.Warn("mana debit failed (operator reconcile)",
					"user_prefix", safePrefix(userGCID, 8),
					"crew_kind", crewKind,
					"agent_kind", agentKind,
					"err", debitErr)
			}

			// active trace span attribute stamping per ADR-141 D1
			// accountability — every mana charge MUST surface on the
			// active trace span for IMDA D3 evidence audit + per-user
			// cost dashboards.
			balancePost := balancePre
			if debitErr == nil {
				balancePost = balancePre - actualTokens
			}
			EmitBalanceSpan(ic, balancePre, balancePost, crewKind)
		},
	})
}

// ChoraEnv reads CHORA_ENV with "dev" fallback. Exported so crews can
// use it when constructing OTEL resource attributes.
func ChoraEnv() string {
	if v := os.Getenv("CHORA_ENV"); v != "" {
		return v
	}
	return "dev"
}

func getStateString(state interface {
	Get(string) (any, error)
}, key string) string {
	v, err := state.Get(key)
	if err != nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

func safePrefix(s string, n int) string {
	if len(s) < n {
		return s
	}
	return s[:n] + "…"
}

// getStateInt reads an int from state. Returns 0 on miss or type
// mismatch — caller treats 0 as "unknown".
func getStateInt(state interface {
	Get(string) (any, error)
}, key string) int {
	v, err := state.Get(key)
	if err != nil {
		return 0
	}
	switch x := v.(type) {
	case int:
		return x
	case int64:
		return int(x)
	case float64:
		return int(x)
	default:
		return 0
	}
}

// BalanceSpanAttributes returns the canonical active trace span attributes
// for a mana debit. `pre` is the balance BEFORE the call (from PeekBalance)
// + `post` is the balance AFTER the post-call Debit. Charge = pre - post,
// clamped to 0 for defensive legibility (a refund or accounting bug would
// otherwise show as a negative).
//
// Used by EmitBalanceSpan + exposed publicly so callers (e.g., LangGraph
// orchestrators bridging via gRPC) can emit equivalent attributes on
// their own spans for cross-process trace correlation.
func BalanceSpanAttributes(pre, post int, crewKind string) []attribute.KeyValue {
	charge := pre - post
	if charge < 0 {
		charge = 0
	}
	return []attribute.KeyValue{
		attribute.Int64("chora.mana.charged", int64(charge)),
		attribute.Int64("chora.mana.balance_post", int64(post)),
		attribute.String("chora.crew_kind", crewKind),
	}
}

// EmitBalanceSpan stamps the canonical mana-charge attributes onto the
// active span in ctx. Safe to call when no span is active (no-op).
// Called by manaplugin's AfterRunCallback per ADR-141 D1 accountability.
func EmitBalanceSpan(ctx context.Context, pre, post int, crewKind string) {
	span := trace.SpanFromContext(ctx)
	if span == nil || !span.IsRecording() {
		return
	}
	span.SetAttributes(BalanceSpanAttributes(pre, post, crewKind)...)
}
