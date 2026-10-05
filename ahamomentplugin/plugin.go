// Package ahamomentplugin implements the ADR-149 §"Source revelation"
// Stage-3 Aha-moment mechanic: when a Familiar first reaches Stage 3,
// it gets a one-shot 24-hour preview of Stage-6 capabilities.
//
// The window is bounded by `aha_moment_active_until` (time.Time on
// session.State, set by chora-consumption.FamiliarHandler at session
// creation if the Familiar's first-Stage-3 transition is in window).
//
// While in window AND growth_stage == 3, this plugin OVERRIDES the keys
// set by `growthstageplugin`:
//
//   - growth.allowed_tools  -> Stage-6 superset
//   - growth.memory_mode    -> vector-rag-all
//   - growth.aha_active     -> true (flag for downstream plugins +
//     builder.ComposeInstruction to inject
//     the "preview" copy)
//
// Outside the window (or wrong stage), the plugin is a no-op — the
// keys set by growthstageplugin pass through unchanged.
//
// Per ADR-149: the LLM-tier preview lift is handled by
// `tieredmodelplugin`. Since F-G4-1 (2026-05-13) tieredmodelplugin
// computes aha_active inline from `aha_moment_active_until` rather
// than reading `growth.aha_active`, because the hosted ADK runtime
// does NOT propagate BeforeRunCallback state writes to
// BeforeModelCallback's ReadonlyState. This plugin still writes
// `growth.aha_active` for local-dev / future ADK versions where the
// state-snapshot bug is fixed.
//
// Plugin chain order (from cmd/familiar/main.go):
//
//	manaplugin → instancedispatch → growthstageplugin → ahamomentplugin →
//	tieredmodelplugin → terminationplugin
//
// ahamomentplugin MUST run AFTER growthstageplugin (consumes its
// allowed_tools / memory_mode keys). Its BEFORE-relationship with
// tieredmodelplugin is no longer load-bearing for the model-tier lift.
package ahamomentplugin

import (
	"errors"
	"fmt"
	"log/slog"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/plugin"

	"github.com/apollo-chora/chora-adk-common/growthstageplugin"
)

// State keys (kept in sync with growthstageplugin where appropriate).
const (
	// Input keys (caller writes these):
	StateKeyGrowthStage = growthstageplugin.StateKeyGrowthStage
	// AhaMomentActiveUntilKey is the timestamp (time.Time, RFC3339 string, or
	// time.Time pointer) marking the END of the 24h Aha-moment preview window.
	AhaMomentActiveUntilKey = "aha_moment_active_until"

	// Output keys (plugin writes these on override):
	StateKeyAllowedTools = growthstageplugin.StateKeyAllowedTools
	StateKeyMemoryMode   = growthstageplugin.StateKeyMemoryMode
	StateKeyAhaActive    = "growth.aha_active"
)

// AhaMomentStage is the stage at which the Aha-moment mechanic triggers.
// Per ADR-149 §"Stage 3 Source revelation".
const AhaMomentStage = 3

// AhaMomentMemoryMode is the memory-mode lift granted by an active Aha moment.
const AhaMomentMemoryMode = "vector-rag-all"

// Config configures the plugin for a specific crew.
type Config struct {
	// CrewKind identifies the owning crew. REQUIRED.
	CrewKind string

	// Now is a clock function for tests. Defaults to time.Now.
	Now func() time.Time
}

// New constructs the Aha-moment plugin.
func New(cfg Config) (*plugin.Plugin, error) {
	if cfg.CrewKind == "" {
		return nil, errors.New(
			"ahamomentplugin: CrewKind required (multi-crew safety — every " +
				"crew must self-identify for IMDA D1 accountability)")
	}
	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	name := fmt.Sprintf("chora_aha_moment_%s", cfg.CrewKind)
	return plugin.New(plugin.Config{
		Name: name,
		BeforeRunCallback: func(ic agent.InvocationContext) (*genai.Content, error) {
			state := ic.Session().State()
			if _, err := ApplyAhaOverride(state, now()); err != nil {
				return nil, fmt.Errorf("ahamomentplugin: %w", err)
			}
			return nil, nil
		},
	})
}

// stateReadWriter is the minimal session.State surface we need.
type stateReadWriter interface {
	Get(key string) (any, error)
	Set(key string, value any) error
}

// ApplyAhaOverride is the pure-function core of BeforeRunCallback. Reads
// `growth_stage` + `aha_moment_active_until` from state and, if the window
// is active for Stage 3, overrides `growth.allowed_tools`, `growth.memory_mode`,
// and sets `growth.aha_active=true`.
//
// Returns (active, nil) where `active=true` means the override fired.
// Returns (false, nil) when the plugin is a no-op (wrong stage, window
// missing, or window expired). Returns (false, err) on table-lookup failure.
func ApplyAhaOverride(state stateReadWriter, now time.Time) (bool, error) {
	stage, ok := readIntState(state, StateKeyGrowthStage)
	if !ok {
		// growthstageplugin already defaulted to Stage 0; absence here is fine.
		return false, nil
	}
	if stage != AhaMomentStage {
		// Defensive: window-active but stage != 3 must be a no-op. Preserves
		// the ADR-149 invariant that Aha-moment is a Stage-3 mechanic.
		return false, nil
	}

	until, ok := readTimeState(state, AhaMomentActiveUntilKey)
	if !ok {
		return false, nil
	}
	if !now.Before(until) {
		// Window expired.
		return false, nil
	}

	// Window active: override. The lift advertises only SHIPPED stage-6
	// ladder names (ADR-249 A1a): an aha preview must not direct the model
	// at tools that do not exist any more than the base stage may.
	stage6Tools, err := growthstageplugin.ShippedStageTools(6)
	if err != nil {
		return false, fmt.Errorf("resolve stage 6 caps: %w", err)
	}
	if err := state.Set(StateKeyAllowedTools, stage6Tools); err != nil {
		slog.Warn("ahamomentplugin: set allowed_tools failed", "err", err)
	}
	if err := state.Set(StateKeyMemoryMode, AhaMomentMemoryMode); err != nil {
		slog.Warn("ahamomentplugin: set memory_mode failed", "err", err)
	}
	if err := state.Set(StateKeyAhaActive, true); err != nil {
		slog.Warn("ahamomentplugin: set aha_active failed", "err", err)
	}
	return true, nil
}

func readIntState(state stateReadWriter, key string) (int, bool) {
	raw, err := state.Get(key)
	if err != nil {
		return 0, false
	}
	switch v := raw.(type) {
	case int:
		return v, true
	case int32:
		return int(v), true
	case int64:
		return int(v), true
	case float64:
		return int(v), true
	default:
		return 0, false
	}
}

func readTimeState(state stateReadWriter, key string) (time.Time, bool) {
	raw, err := state.Get(key)
	if err != nil {
		return time.Time{}, false
	}
	switch v := raw.(type) {
	case time.Time:
		return v, true
	case *time.Time:
		if v == nil {
			return time.Time{}, false
		}
		return *v, true
	case string:
		// Try RFC3339 first (canonical event timestamp format), then
		// RFC3339Nano for nanosecond precision.
		if t, err := time.Parse(time.RFC3339, v); err == nil {
			return t, true
		}
		if t, err := time.Parse(time.RFC3339Nano, v); err == nil {
			return t, true
		}
		return time.Time{}, false
	default:
		return time.Time{}, false
	}
}
