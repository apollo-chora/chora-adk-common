package growthstageplugin

import (
	_ "embed"
	"fmt"
	"sync"

	"gopkg.in/yaml.v3"
)

//go:embed config.yaml
var defaultYAML []byte

// configFile mirrors the on-disk YAML layout.
type configFile struct {
	// UnshippedTools declares ladder names granted by the stage table whose
	// backing agent tools do not ship (ADR-249 A1a). Optional; empty means
	// every granted name must map in LadderToRegistryKey for Default().
	UnshippedTools []string `yaml:"unshipped_tools"`
	Stages         []struct {
		Stage        int      `yaml:"stage"`
		Name         string   `yaml:"name"`
		AllowedTools []string `yaml:"allowed_tools"`
		MemoryMode   string   `yaml:"memory_mode"`
	} `yaml:"stages"`
}

var (
	defaultTableOnce sync.Once
	defaultTable     *CapabilityTable
	defaultTableErr  error
)

// Default returns the process-lifetime cached CapabilityTable parsed from the
// embedded config.yaml.
//
// Cached at first call. Deterministic + reproducible per IMDA D2 transparency.
func Default() (*CapabilityTable, error) {
	defaultTableOnce.Do(func() {
		defaultTable, defaultTableErr = LoadFromYAML(defaultYAML)
		if defaultTableErr == nil {
			// ADR-249 A1a agree-invariant, enforced at load so a vocabulary
			// drift fails the process rather than degrading into the
			// narrowing-safe WARN: every granted ladder name is either
			// mapped to a shipped registry key or declared unshipped.
			defaultTableErr = validateLadderShippable(defaultTable)
		}
	})
	return defaultTable, defaultTableErr
}

// validateLadderShippable enforces that every ladder name in the table is
// either a key of LadderToRegistryKey or declared in unshipped_tools.
func validateLadderShippable(t *CapabilityTable) error {
	for stage := 0; stage <= 6; stage++ {
		caps, ok := t.byStage[stage]
		if !ok {
			continue
		}
		for _, name := range caps.AllowedTools {
			if t.unshipped[name] {
				continue
			}
			if _, mapped := LadderToRegistryKey[name]; !mapped {
				return fmt.Errorf(
					"growthstageplugin: stage %d ladder name %q neither maps to a "+
						"shipped registry key nor is declared unshipped (ADR-249 A1a)",
					stage, name)
			}
		}
	}
	return nil
}

// LoadFromYAML parses a CapabilityTable from a YAML payload. Exported so
// tests + operators can supply a custom config file (e.g., for per-tenant
// stage tuning at sandbox time).
//
// Validates:
//   - exactly 7 stages (0-6) — every stage required
//   - no duplicate stages
//   - allowed_tools slices are non-nil (empty list allowed for Stage 0)
//   - tool list monotonicity (stage N ⊇ stage N-1)
//
// Monotonicity is a NON-NEGOTIABLE invariant per ADR-149 ("growth is
// monotonic — no demotion"). A YAML drift that drops a tool at a higher
// stage = ADR violation, refused at load time.
func LoadFromYAML(raw []byte) (*CapabilityTable, error) {
	var cf configFile
	if err := yaml.Unmarshal(raw, &cf); err != nil {
		return nil, fmt.Errorf("growthstageplugin: parse config yaml: %w", err)
	}
	if len(cf.Stages) != 7 {
		return nil, fmt.Errorf(
			"growthstageplugin: want 7 stages (0-6); got %d", len(cf.Stages))
	}
	t := &CapabilityTable{
		byStage:   make(map[int]StageCaps, 7),
		unshipped: make(map[string]bool, len(cf.UnshippedTools)),
	}
	for _, u := range cf.UnshippedTools {
		t.unshipped[u] = true
	}
	for _, s := range cf.Stages {
		if s.Stage < 0 || s.Stage > 6 {
			return nil, fmt.Errorf(
				"growthstageplugin: stage %d out of range (0-6)", s.Stage)
		}
		if _, dup := t.byStage[s.Stage]; dup {
			return nil, fmt.Errorf(
				"growthstageplugin: duplicate stage %d in config", s.Stage)
		}
		if s.MemoryMode == "" {
			return nil, fmt.Errorf(
				"growthstageplugin: stage %d missing memory_mode", s.Stage)
		}
		tools := s.AllowedTools
		if tools == nil {
			tools = []string{}
		}
		t.byStage[s.Stage] = StageCaps{
			Stage:        s.Stage,
			Name:         s.Name,
			AllowedTools: tools,
			MemoryMode:   s.MemoryMode,
		}
	}
	// Verify every stage 0-6 is present.
	for i := 0; i <= 6; i++ {
		if _, ok := t.byStage[i]; !ok {
			return nil, fmt.Errorf(
				"growthstageplugin: missing stage %d in config", i)
		}
	}
	// Monotonicity check: stage N's tool set ⊇ stage N-1's.
	for i := 1; i <= 6; i++ {
		prev := t.byStage[i-1].AllowedTools
		curr := t.byStage[i].AllowedTools
		for _, p := range prev {
			found := false
			for _, c := range curr {
				if c == p {
					found = true
					break
				}
			}
			if !found {
				return nil, fmt.Errorf(
					"growthstageplugin: stage %d dropped tool %q from stage %d "+
						"(ADR-149 monotonicity invariant)", i, p, i-1)
			}
		}
	}
	// A stale unshipped declaration (naming a tool no stage grants) is a
	// config bug: it reads as coverage of a name that does not exist.
	for name := range t.unshipped {
		granted := false
		for _, caps := range t.byStage {
			for _, n := range caps.AllowedTools {
				if n == name {
					granted = true
					break
				}
			}
			if granted {
				break
			}
		}
		if !granted {
			return nil, fmt.Errorf(
				"growthstageplugin: unshipped_tools names %q, which no stage grants "+
					"(stale declaration)", name)
		}
	}
	return t, nil
}
