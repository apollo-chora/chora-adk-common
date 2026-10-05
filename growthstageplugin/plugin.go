// Package growthstageplugin reads the per-Familiar `growth_stage` (int 0-6)
// + `species` (string) from session.State and gates the agent's capability
// surface per ADR-149 §"The 7 stages":
//
//   - growth.allowed_tools        — []string  (gated tool name list)
//   - growth.memory_mode          — string    (stateless | rolling-N | ...)
//   - growth.kg_neighbor_atom_ids — []string  (from visible_kg_neighbors)
//
// Upstream callers (e.g., chora-consumption.FamiliarHandler at session
// `async_create_session` time) MUST set `growth_stage` + `species` (and
// optionally `visible_kg_neighbors`) on session.State. A missing
// `growth_stage` defaults to stage 0 (Egg, the safest pre-hatch posture).
//
// Plugin chain order (from cmd/familiar/main.go):
//
//	manaplugin → instancedispatch → growthstageplugin → ahamomentplugin →
//	tieredmodelplugin → terminationplugin
//
// growthstageplugin MUST run AFTER instancedispatch (which loads the
// FamiliarConfig from the registry and propagates growth_stage / species)
// and BEFORE ahamomentplugin + tieredmodelplugin (which read the gated keys
// this plugin writes). The Aha-moment plugin may OVERRIDE these keys when
// the 24-hour Source-revelation preview is active.
//
// Per `feedback_no_inline_config`: the per-stage gating table is loaded
// from `config.yaml` (embedded via go:embed) — operator-tunable without
// rebuilding the binary.
package growthstageplugin

import (
	"errors"
	"fmt"
	"log/slog"
	"sort"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/plugin"
)

// Session-state keys.
const (
	// Inputs (caller must set at session creation):
	StateKeyGrowthStage        = "growth_stage"
	StateKeySpecies            = "species"
	StateKeyVisibleKgNeighbors = "visible_kg_neighbors"

	// Outputs (plugin writes these for downstream plugins + builder):
	StateKeyAllowedTools     = "growth.allowed_tools"
	StateKeyMemoryMode       = "growth.memory_mode"
	StateKeyKgNeighborAtomID = "growth.kg_neighbor_atom_ids"
	StateKeyResolvedSpecies  = "growth.species"
)

// Config configures the plugin for a specific crew.
type Config struct {
	// CrewKind identifies the owning crew (e.g., "familiar"). REQUIRED.
	CrewKind string

	// Table overrides the default capability table. Empty -> embedded default.
	Table *CapabilityTable
}

// CapabilityTable maps growth_stage (0-6) -> StageCaps.
type CapabilityTable struct {
	byStage map[int]StageCaps
	// unshipped holds ladder names granted by the stage table whose backing
	// agent tools do not ship (ADR-249 A1a). Resolve() keeps returning the
	// RAW ladder (it is the cross-service alignment surface against
	// chora-consumption growth.toolsByStage); the runtime state write goes
	// through ShippedTools() so the agent never advertises an unbacked name.
	unshipped map[string]bool
}

// LadderToRegistryKey maps each SHIPPED ladder name (ADR-218 vocabulary) to
// the agent tool-registry key that backs it (ADR-249 A1a agree-invariant).
// Every ladder name in config.yaml must appear here or in unshipped_tools;
// Default() refuses the config otherwise, and cmd/familiar asserts at boot
// that every value below is a key of its availableTools registry.
var LadderToRegistryKey = map[string]string{
	"cite_atom": "atom.cite",
}

// Unshipped returns the declared-unshipped ladder names, sorted.
func (t *CapabilityTable) Unshipped() []string {
	if t == nil {
		return nil
	}
	out := make([]string, 0, len(t.unshipped))
	for name := range t.unshipped {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// ShippedTools returns the stage's ladder names minus the declared-unshipped
// set: the list safe to advertise at runtime.
func (t *CapabilityTable) ShippedTools(stage int) ([]string, error) {
	caps, err := t.Resolve(stage)
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(caps.AllowedTools))
	for _, name := range caps.AllowedTools {
		if !t.unshipped[name] {
			out = append(out, name)
		}
	}
	return out, nil
}

// ShippedStageTools is the package-level convenience over the default table.
func ShippedStageTools(stage int) ([]string, error) {
	table, err := Default()
	if err != nil {
		return nil, err
	}
	return table.ShippedTools(stage)
}

// StageCaps is the per-stage capability gate.
type StageCaps struct {
	Stage        int
	Name         string
	AllowedTools []string
	MemoryMode   string
}

// New constructs the growth-stage gating plugin.
func New(cfg Config) (*plugin.Plugin, error) {
	if cfg.CrewKind == "" {
		return nil, errors.New(
			"growthstageplugin: CrewKind required (multi-crew safety — every " +
				"crew must self-identify for IMDA D1 accountability)")
	}
	table := cfg.Table
	if table == nil {
		var err error
		table, err = Default()
		if err != nil {
			return nil, fmt.Errorf("growthstageplugin: load default table: %w", err)
		}
	}
	name := fmt.Sprintf("chora_growth_stage_%s", cfg.CrewKind)
	return plugin.New(plugin.Config{
		Name: name,
		BeforeRunCallback: func(ic agent.InvocationContext) (*genai.Content, error) {
			return applyStageGating(ic, table)
		},
	})
}

func applyStageGating(ic agent.InvocationContext, table *CapabilityTable) (*genai.Content, error) {
	state := ic.Session().State()

	stage, ok, err := readStageFromState(state)
	if err != nil {
		return nil, err
	}
	if !ok {
		// Missing growth_stage -> default to Stage 0 (Egg). Egg is the
		// safest posture: no tools, stateless memory. Caller can recover
		// by setting growth_stage at the next session_create.
		stage = 0
		slog.Debug("growthstageplugin: growth_stage missing; defaulting to Stage 0 (Egg)")
	}

	caps, err := table.Resolve(stage)
	if err != nil {
		return nil, fmt.Errorf("growthstageplugin: %w", err)
	}

	// ADR-249 A1a: advertise only ladder names whose tools ship. The raw
	// ladder (caps.AllowedTools) stays the product-design surface; the
	// runtime allowlist must not direct the model at an unbacked name.
	shipped, err := table.ShippedTools(stage)
	if err != nil {
		return nil, fmt.Errorf("growthstageplugin: %w", err)
	}
	if err := state.Set(StateKeyAllowedTools, shipped); err != nil {
		slog.Warn("growthstageplugin: set allowed_tools failed", "err", err)
	}
	if err := state.Set(StateKeyMemoryMode, caps.MemoryMode); err != nil {
		slog.Warn("growthstageplugin: set memory_mode failed", "err", err)
	}

	species := readStringState(state, StateKeySpecies)
	if species != "" {
		if err := state.Set(StateKeyResolvedSpecies, species); err != nil {
			slog.Warn("growthstageplugin: set species failed", "err", err)
		}
	}

	neighbors := readStringSliceState(state, StateKeyVisibleKgNeighbors)
	// Always set the output key — even empty — so downstream plugins +
	// the builder can rely on the key existing.
	if err := state.Set(StateKeyKgNeighborAtomID, append([]string(nil), neighbors...)); err != nil {
		slog.Warn("growthstageplugin: set kg_neighbors failed", "err", err)
	}

	return nil, nil
}

// readStageFromState reads growth_stage from state, accepting int / int64 /
// float64 (JSON-unmarshal compatibility). Returns:
//
//	(stage, true, nil)   — present + valid type
//	(0, false, nil)      — absent
//	(0, false, error)    — present but wrong type
func readStageFromState(state interface {
	Get(string) (any, error)
}) (int, bool, error) {
	raw, err := state.Get(StateKeyGrowthStage)
	if err != nil {
		// session.ErrStateKeyNotExist returned by all known State impls —
		// treat any error from Get as "not present".
		return 0, false, nil
	}
	switch v := raw.(type) {
	case int:
		return v, true, nil
	case int32:
		return int(v), true, nil
	case int64:
		return int(v), true, nil
	case float64:
		return int(v), true, nil
	default:
		return 0, false, fmt.Errorf("growth_stage must be numeric; got %T", raw)
	}
}

func readStringState(state interface {
	Get(string) (any, error)
}, key string) string {
	raw, err := state.Get(key)
	if err != nil {
		return ""
	}
	if s, ok := raw.(string); ok {
		return s
	}
	return ""
}

func readStringSliceState(state interface {
	Get(string) (any, error)
}, key string) []string {
	raw, err := state.Get(key)
	if err != nil {
		return nil
	}
	switch v := raw.(type) {
	case []string:
		return v
	case []any:
		out := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}

// Resolve returns the StageCaps for a given growth_stage. Returns error for
// stages outside 0-6 inclusive.
func (t *CapabilityTable) Resolve(stage int) (StageCaps, error) {
	if t == nil || t.byStage == nil {
		return StageCaps{}, errors.New("growthstageplugin: capability table is empty")
	}
	caps, ok := t.byStage[stage]
	if !ok {
		return StageCaps{}, fmt.Errorf("growth_stage %d out of range (expected 0-6)", stage)
	}
	// Defensive copy of the AllowedTools slice — callers must not mutate.
	out := caps
	out.AllowedTools = append([]string(nil), caps.AllowedTools...)
	return out, nil
}

// ResolveStageCaps is a package-level convenience wrapper around the default
// table for callers who don't hold a CapabilityTable handle (mostly tests +
// downstream plugins that don't need to override the table).
func ResolveStageCaps(stage int) (StageCaps, error) {
	table, err := Default()
	if err != nil {
		return StageCaps{}, err
	}
	return table.Resolve(stage)
}
