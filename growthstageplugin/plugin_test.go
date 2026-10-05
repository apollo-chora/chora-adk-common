package growthstageplugin

import (
	"context"
	"iter"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// growthstageplugin reads `growth_stage` (int 0-6) + `species` (string) from
// session.State and sets gating keys (`growth.allowed_tools`, `growth.memory_mode`,
// `growth.kg_neighbor_atom_ids`) per ADR-149 §"The 7 stages" table.

// ---------- New() validation ----------

func TestNew_rejectsEmptyCrewKind(t *testing.T) {
	_, err := New(Config{})
	if err == nil {
		t.Fatal("want error for empty CrewKind; got nil")
	}
	if !strings.Contains(err.Error(), "CrewKind") {
		t.Errorf("error should mention CrewKind; got %v", err)
	}
}

func TestNew_buildsPluginWithExpectedName(t *testing.T) {
	p, err := New(Config{CrewKind: "familiar"})
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if !strings.Contains(p.Name(), "growth_stage") {
		t.Errorf("plugin name should describe purpose; got %q", p.Name())
	}
	if !strings.Contains(p.Name(), "familiar") {
		t.Errorf("plugin name should embed CrewKind; got %q", p.Name())
	}
}

func TestNew_wiresBeforeRunCallback(t *testing.T) {
	p, err := New(Config{CrewKind: "familiar"})
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if p.BeforeRunCallback() == nil {
		t.Fatal("BeforeRunCallback must be wired")
	}
}

// ---------- ResolveStageCaps (pure function) ----------

func TestResolveStageCaps_stage0_egg(t *testing.T) {
	caps, err := ResolveStageCaps(0)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if len(caps.AllowedTools) != 0 {
		t.Errorf("stage 0 (Egg): want 0 allowed tools; got %v", caps.AllowedTools)
	}
	if caps.MemoryMode != "stateless" {
		t.Errorf("stage 0: want memory_mode=stateless; got %q", caps.MemoryMode)
	}
}

func TestResolveStageCaps_stage1_baby(t *testing.T) {
	caps, err := ResolveStageCaps(1)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if !contains(caps.AllowedTools, "cite_atom") {
		t.Errorf("stage 1 (Baby): cite_atom must be allowed; got %v", caps.AllowedTools)
	}
	if caps.MemoryMode != "rolling-1" {
		t.Errorf("stage 1: want memory_mode=rolling-1; got %q", caps.MemoryMode)
	}
}

func TestResolveStageCaps_stage2_fledgling(t *testing.T) {
	caps, err := ResolveStageCaps(2)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if !contains(caps.AllowedTools, "cite_atom") || !contains(caps.AllowedTools, "atom_search") {
		t.Errorf("stage 2 (Fledgling): cite_atom + atom_search must be allowed; got %v", caps.AllowedTools)
	}
	if caps.MemoryMode != "rolling-3" {
		t.Errorf("stage 2: want memory_mode=rolling-3; got %q", caps.MemoryMode)
	}
}

func TestResolveStageCaps_stage3_awakened(t *testing.T) {
	caps, err := ResolveStageCaps(3)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	wantTools := []string{"cite_atom", "atom_search", "ebbinghaus_state"}
	for _, n := range wantTools {
		if !contains(caps.AllowedTools, n) {
			t.Errorf("stage 3 (Awakened): %s must be allowed; got %v", n, caps.AllowedTools)
		}
	}
	if caps.MemoryMode != "rolling-7" {
		t.Errorf("stage 3: want memory_mode=rolling-7; got %q", caps.MemoryMode)
	}
}

func TestResolveStageCaps_stage4_structural(t *testing.T) {
	caps, err := ResolveStageCaps(4)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	// ADR-218 D1 spec §1: persona_lookup AND score_atom_for_learner both arrive
	// at st4 (score_atom_for_learner moved down from the old st5 placement).
	for _, n := range []string{"persona_lookup", "score_atom_for_learner"} {
		if !contains(caps.AllowedTools, n) {
			t.Errorf("stage 4 (Structural): %s must be allowed; got %v", n, caps.AllowedTools)
		}
	}
	if caps.MemoryMode != "rolling-21-rag" {
		t.Errorf("stage 4: want memory_mode=rolling-21-rag; got %q", caps.MemoryMode)
	}
}

func TestResolveStageCaps_stage5_teen(t *testing.T) {
	caps, err := ResolveStageCaps(5)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	// ADR-218 D1: stage 5 adds NO new innate tools (the kit plateaus at st4);
	// its gain is memory reach. query_kg is RETIRED to the map_sight Skill.
	if !contains(caps.AllowedTools, "score_atom_for_learner") {
		t.Errorf("stage 5 (Teen): score_atom_for_learner must be allowed; got %v", caps.AllowedTools)
	}
	if contains(caps.AllowedTools, "query_kg") {
		t.Errorf("stage 5 (Teen): query_kg is RETIRED (→ map_sight); must not appear; got %v", caps.AllowedTools)
	}
	if caps.MemoryMode != "vector-rag-all" {
		t.Errorf("stage 5: want memory_mode=vector-rag-all; got %q", caps.MemoryMode)
	}
}

func TestResolveStageCaps_stage6_matured(t *testing.T) {
	caps, err := ResolveStageCaps(6)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	// ADR-218 D1: the innate meta-tools are RETIRED — suggest_atom_authoring →
	// atom_forge Skill; propose_kg_merge → fog_scout's st6 extension. Stage 6
	// adds only full-procedural memory over the plateaued st4 kit.
	retiredMeta := []string{"suggest_atom_authoring", "propose_kg_merge"}
	for _, n := range retiredMeta {
		if contains(caps.AllowedTools, n) {
			t.Errorf("stage 6 (Matured): %s is RETIRED to the catalogue; must not appear; got %v", n, caps.AllowedTools)
		}
	}
	if !contains(caps.AllowedTools, "score_atom_for_learner") {
		t.Errorf("stage 6 (Matured): plateaued st4 kit must persist; got %v", caps.AllowedTools)
	}
	if caps.MemoryMode != "full-procedural" {
		t.Errorf("stage 6: want memory_mode=full-procedural; got %q", caps.MemoryMode)
	}
}

func TestResolveStageCaps_monotonicallyExpandsTools(t *testing.T) {
	// Stages MUST be monotonic in capability: stage N's tool set must be a
	// superset (or equal) of stage N-1's tools per ADR-149 ("growth is
	// monotonic — no demotion").
	prev := []string{}
	for s := 1; s <= 6; s++ {
		caps, err := ResolveStageCaps(s)
		if err != nil {
			t.Fatalf("stage %d: %v", s, err)
		}
		for _, want := range prev {
			if !contains(caps.AllowedTools, want) {
				t.Errorf("stage %d dropped tool %q (must be monotonic); got %v",
					s, want, caps.AllowedTools)
			}
		}
		prev = caps.AllowedTools
	}
}

// canonicalInnateLadder is the ALIGNED innate-kit ladder (ADR-218 D1 + spec
// pack docs/FAMILIAR-SKILL-SPECS-2026-07-03.md §1), kept set-identical to the
// canonical chora-consumption growth.toolsByStage (curve.go). The kit PLATEAUS
// at stage 4 — every higher-order power is now an equippable catalogue Skill
// occupying a slot, NOT an innate tool. The embedded config.yaml is graded
// against this map; drift in either fails loud (CHO-2014 P0-debt alignment).
var canonicalInnateLadder = map[int][]string{
	0: {},
	1: {"cite_atom"},
	2: {"cite_atom", "atom_search"},
	3: {"cite_atom", "atom_search", "ebbinghaus_state"},
	4: {"cite_atom", "atom_search", "ebbinghaus_state", "persona_lookup", "score_atom_for_learner"},
	5: {"cite_atom", "atom_search", "ebbinghaus_state", "persona_lookup", "score_atom_for_learner"},
	6: {"cite_atom", "atom_search", "ebbinghaus_state", "persona_lookup", "score_atom_for_learner"},
}

// retiredInnateTools are the ADR-149-era ladder names moved OUT of the innate
// kit into the equippable catalogue by ADR-218: query_kg → map_sight
// (kg.read_map), suggest_atom_authoring → atom_forge, propose_kg_merge →
// fog_scout's st6 extension. They must never appear at any innate stage.
var retiredInnateTools = []string{"query_kg", "suggest_atom_authoring", "propose_kg_merge"}

// TestResolveStageCaps_matchesCanonicalInnateLadder is the P0-debt alignment
// gate (CHO-2014): the growthstageplugin config.yaml must match the canonical
// consumption ladder EXACTLY at every stage, and must carry none of the retired
// meta-tools. This is the RED assertion the pre-alignment YAML fails.
func TestResolveStageCaps_matchesCanonicalInnateLadder(t *testing.T) {
	for stage := 0; stage <= 6; stage++ {
		caps, err := ResolveStageCaps(stage)
		if err != nil {
			t.Fatalf("stage %d: unexpected err %v", stage, err)
		}
		want := canonicalInnateLadder[stage]
		if !sameStringSet(caps.AllowedTools, want) {
			t.Errorf("stage %d innate kit mismatch:\n  got  %v\n  want %v", stage, caps.AllowedTools, want)
		}
		for _, r := range retiredInnateTools {
			if contains(caps.AllowedTools, r) {
				t.Errorf("stage %d still lists RETIRED innate tool %q (ADR-218 moved it to the catalogue)", stage, r)
			}
		}
	}
}

func TestResolveStageCaps_rejectsInvalidStage(t *testing.T) {
	for _, stage := range []int{-1, 7, 100} {
		_, err := ResolveStageCaps(stage)
		if err == nil {
			t.Errorf("stage %d should error; got nil", stage)
		}
	}
}

// ---------- BeforeRunCallback integration ----------

func TestBeforeRunCallback_setsAllowedToolsAndMemoryMode(t *testing.T) {
	p, _ := New(Config{CrewKind: "familiar"})

	state := &fakeState{m: map[string]any{"growth_stage": 4, "species": "fox"}}
	ic := newFakeInvocationCtx(state)

	if _, err := p.BeforeRunCallback()(ic); err != nil {
		t.Fatalf("BeforeRunCallback err: %v", err)
	}
	got, _ := state.Get("growth.allowed_tools")
	tools, ok := got.([]string)
	if !ok {
		t.Fatalf("growth.allowed_tools must be []string; got %T", got)
	}
	// ADR-249 A1a: the runtime allowlist advertises only SHIPPED ladder
	// names. Stage 4's raw ladder still grants persona_lookup (see
	// TestResolveStageCaps_stage4_structural), but its tool does not ship,
	// so the state write filters it.
	if !sameStringSet(tools, []string{"cite_atom"}) {
		t.Errorf("stage 4 runtime allowlist must be exactly [cite_atom] (shipped only); got %v", tools)
	}
	if contains(tools, "persona_lookup") {
		t.Errorf("stage 4 runtime allowlist must not advertise unshipped persona_lookup; got %v", tools)
	}
	mm, _ := state.Get("growth.memory_mode")
	if mm != "rolling-21-rag" {
		t.Errorf("stage 4 memory_mode mismatch; got %v", mm)
	}
}

// ---------- ADR-249 A1a: unshipped declarations ----------

func TestDefaultTable_declaresUnshippedLadderNames(t *testing.T) {
	table, err := Default()
	if err != nil {
		t.Fatalf("Default: %v", err)
	}
	un := table.Unshipped()
	want := []string{"atom_search", "ebbinghaus_state", "persona_lookup", "score_atom_for_learner"}
	if !sameStringSet(un, want) {
		t.Errorf("unshipped declarations = %v, want %v", un, want)
	}
}

func TestShippedTools_filtersUnshippedNames(t *testing.T) {
	table, err := Default()
	if err != nil {
		t.Fatalf("Default: %v", err)
	}
	for stage := 1; stage <= 6; stage++ {
		shipped, err := table.ShippedTools(stage)
		if err != nil {
			t.Fatalf("stage %d: %v", stage, err)
		}
		if !sameStringSet(shipped, []string{"cite_atom"}) {
			t.Errorf("stage %d shipped tools = %v, want exactly [cite_atom]", stage, shipped)
		}
	}
	shipped0, err := table.ShippedTools(0)
	if err != nil {
		t.Fatalf("stage 0: %v", err)
	}
	if len(shipped0) != 0 {
		t.Errorf("stage 0 (Egg) shipped tools must be empty; got %v", shipped0)
	}
}

func TestResolveStageCaps_rawLadderStillCarriesUnshippedNames(t *testing.T) {
	// Resolve() is the cross-service alignment surface against
	// chora-consumption growth.toolsByStage and must keep returning the RAW
	// ladder; only the runtime state write filters.
	caps, err := ResolveStageCaps(4)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if !contains(caps.AllowedTools, "persona_lookup") {
		t.Errorf("raw stage 4 ladder must keep persona_lookup (product design surface); got %v", caps.AllowedTools)
	}
}

func TestLoadFromYAML_rejectsStaleUnshippedDeclaration(t *testing.T) {
	raw := []byte(`unshipped_tools:
  - never_granted_anywhere
stages:
  - {stage: 0, name: egg, allowed_tools: [], memory_mode: stateless}
  - {stage: 1, name: a, allowed_tools: [cite_atom], memory_mode: rolling-1}
  - {stage: 2, name: b, allowed_tools: [cite_atom], memory_mode: rolling-3}
  - {stage: 3, name: c, allowed_tools: [cite_atom], memory_mode: rolling-7}
  - {stage: 4, name: d, allowed_tools: [cite_atom], memory_mode: rolling-21-rag}
  - {stage: 5, name: e, allowed_tools: [cite_atom], memory_mode: vector-rag-all}
  - {stage: 6, name: f, allowed_tools: [cite_atom], memory_mode: full-procedural}
`)
	_, err := LoadFromYAML(raw)
	if err == nil {
		t.Fatal("an unshipped declaration naming a never-granted tool must be refused (stale declaration)")
	}
	if !strings.Contains(err.Error(), "stale") {
		t.Errorf("error should mention the stale declaration; got %v", err)
	}
}

func TestBeforeRunCallback_missingGrowthStage_defaultsToStage0(t *testing.T) {
	p, _ := New(Config{CrewKind: "familiar"})

	state := &fakeState{m: map[string]any{}}
	ic := newFakeInvocationCtx(state)

	if _, err := p.BeforeRunCallback()(ic); err != nil {
		t.Fatalf("missing growth_stage should default to stage 0 (Egg), not error; got: %v", err)
	}
	got, _ := state.Get("growth.allowed_tools")
	tools, ok := got.([]string)
	if !ok {
		t.Fatalf("growth.allowed_tools must be []string; got %T", got)
	}
	if len(tools) != 0 {
		t.Errorf("default stage 0: want empty tool list; got %v", tools)
	}
	mm, _ := state.Get("growth.memory_mode")
	if mm != "stateless" {
		t.Errorf("default stage 0 memory_mode; got %v", mm)
	}
}

func TestBeforeRunCallback_invalidStageReturnsError(t *testing.T) {
	p, _ := New(Config{CrewKind: "familiar"})

	state := &fakeState{m: map[string]any{"growth_stage": 7}}
	ic := newFakeInvocationCtx(state)

	if _, err := p.BeforeRunCallback()(ic); err == nil {
		t.Fatal("invalid stage 7 must return error; got nil")
	}

	state2 := &fakeState{m: map[string]any{"growth_stage": -1}}
	ic2 := newFakeInvocationCtx(state2)
	if _, err := p.BeforeRunCallback()(ic2); err == nil {
		t.Fatal("invalid stage -1 must return error; got nil")
	}
}

func TestBeforeRunCallback_acceptsInt64AndFloat64Stage(t *testing.T) {
	// State.Set may store numeric values as different concrete types
	// depending on caller (JSON unmarshal yields float64, Go literals yield int).
	p, _ := New(Config{CrewKind: "familiar"})

	for _, in := range []any{int(3), int64(3), float64(3)} {
		state := &fakeState{m: map[string]any{"growth_stage": in}}
		ic := newFakeInvocationCtx(state)
		if _, err := p.BeforeRunCallback()(ic); err != nil {
			t.Errorf("growth_stage typed %T should be accepted; got: %v", in, err)
		}
		mm, _ := state.Get("growth.memory_mode")
		if mm != "rolling-7" {
			t.Errorf("typed %T should resolve to stage 3 (rolling-7); got %v", in, mm)
		}
	}
}

func TestBeforeRunCallback_copiesVisibleKgNeighborsToState(t *testing.T) {
	p, _ := New(Config{CrewKind: "familiar"})

	state := &fakeState{m: map[string]any{
		"growth_stage":         4,
		"visible_kg_neighbors": []string{"atom-A", "atom-B"},
	}}
	ic := newFakeInvocationCtx(state)
	if _, err := p.BeforeRunCallback()(ic); err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	got, _ := state.Get("growth.kg_neighbor_atom_ids")
	ids, ok := got.([]string)
	if !ok || len(ids) != 2 || ids[0] != "atom-A" || ids[1] != "atom-B" {
		t.Errorf("growth.kg_neighbor_atom_ids should mirror visible_kg_neighbors; got %v", got)
	}
}

func TestBeforeRunCallback_noVisibleKgNeighborsYieldsEmptySlice(t *testing.T) {
	p, _ := New(Config{CrewKind: "familiar"})

	state := &fakeState{m: map[string]any{"growth_stage": 2}}
	ic := newFakeInvocationCtx(state)
	if _, err := p.BeforeRunCallback()(ic); err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	got, _ := state.Get("growth.kg_neighbor_atom_ids")
	ids, ok := got.([]string)
	if !ok {
		t.Fatalf("growth.kg_neighbor_atom_ids should be []string even when source missing; got %T", got)
	}
	if len(ids) != 0 {
		t.Errorf("expected empty slice; got %v", ids)
	}
}

func TestBeforeRunCallback_acceptsAnySliceForKgNeighbors(t *testing.T) {
	// JSON unmarshal can yield []any rather than []string.
	p, _ := New(Config{CrewKind: "familiar"})
	state := &fakeState{m: map[string]any{
		"growth_stage":         5,
		"visible_kg_neighbors": []any{"atom-1", "atom-2", 123 /* dropped */},
	}}
	ic := newFakeInvocationCtx(state)
	if _, err := p.BeforeRunCallback()(ic); err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	got, _ := state.Get("growth.kg_neighbor_atom_ids")
	ids, ok := got.([]string)
	if !ok || len(ids) != 2 {
		t.Errorf("want 2 strings; got %T %v", got, got)
	}
}

func TestBeforeRunCallback_persistsSpeciesWhenPresent(t *testing.T) {
	p, _ := New(Config{CrewKind: "familiar"})
	state := &fakeState{m: map[string]any{"growth_stage": 1, "species": "owl"}}
	ic := newFakeInvocationCtx(state)
	if _, err := p.BeforeRunCallback()(ic); err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	got, _ := state.Get("growth.species")
	if got != "owl" {
		t.Errorf("want growth.species=owl; got %v", got)
	}
}

func TestBeforeRunCallback_rejectsNonNumericGrowthStage(t *testing.T) {
	p, _ := New(Config{CrewKind: "familiar"})
	state := &fakeState{m: map[string]any{"growth_stage": "not-a-number"}}
	ic := newFakeInvocationCtx(state)
	if _, err := p.BeforeRunCallback()(ic); err == nil {
		t.Fatal("non-numeric growth_stage must error; got nil")
	}
}

func TestBeforeRunCallback_acceptsInt32Stage(t *testing.T) {
	p, _ := New(Config{CrewKind: "familiar"})
	state := &fakeState{m: map[string]any{"growth_stage": int32(6)}}
	ic := newFakeInvocationCtx(state)
	if _, err := p.BeforeRunCallback()(ic); err != nil {
		t.Fatalf("int32 stage should be accepted; got %v", err)
	}
}

// ---------- LoadFromYAML validation paths ----------

func TestLoadFromYAML_rejectsMalformedYAML(t *testing.T) {
	_, err := LoadFromYAML([]byte("not: [valid: yaml"))
	if err == nil {
		t.Fatal("malformed yaml must error")
	}
}

func TestLoadFromYAML_rejectsWrongStageCount(t *testing.T) {
	raw := []byte(`stages:
  - stage: 0
    name: egg
    allowed_tools: []
    memory_mode: stateless
`)
	_, err := LoadFromYAML(raw)
	if err == nil {
		t.Fatal("want 7-stages validation; got nil")
	}
	if !strings.Contains(err.Error(), "7 stages") {
		t.Errorf("error should mention 7-stage requirement; got %v", err)
	}
}

func TestLoadFromYAML_rejectsOutOfRangeStage(t *testing.T) {
	raw := []byte(`stages:
  - {stage: 9, name: x, allowed_tools: [], memory_mode: stateless}
  - {stage: 1, name: a, allowed_tools: [], memory_mode: rolling-1}
  - {stage: 2, name: b, allowed_tools: [], memory_mode: rolling-3}
  - {stage: 3, name: c, allowed_tools: [], memory_mode: rolling-7}
  - {stage: 4, name: d, allowed_tools: [], memory_mode: rolling-21-rag}
  - {stage: 5, name: e, allowed_tools: [], memory_mode: vector-rag-all}
  - {stage: 6, name: f, allowed_tools: [], memory_mode: full-procedural}
`)
	_, err := LoadFromYAML(raw)
	if err == nil {
		t.Fatal("out-of-range stage must error")
	}
}

func TestLoadFromYAML_rejectsDuplicateStage(t *testing.T) {
	raw := []byte(`stages:
  - {stage: 0, name: egg, allowed_tools: [], memory_mode: stateless}
  - {stage: 0, name: dupe, allowed_tools: [], memory_mode: stateless}
  - {stage: 2, name: b, allowed_tools: [], memory_mode: rolling-3}
  - {stage: 3, name: c, allowed_tools: [], memory_mode: rolling-7}
  - {stage: 4, name: d, allowed_tools: [], memory_mode: rolling-21-rag}
  - {stage: 5, name: e, allowed_tools: [], memory_mode: vector-rag-all}
  - {stage: 6, name: f, allowed_tools: [], memory_mode: full-procedural}
`)
	_, err := LoadFromYAML(raw)
	if err == nil {
		t.Fatal("duplicate stage must error")
	}
}

func TestLoadFromYAML_rejectsMissingMemoryMode(t *testing.T) {
	raw := []byte(`stages:
  - {stage: 0, name: egg, allowed_tools: [], memory_mode: ""}
  - {stage: 1, name: a, allowed_tools: [], memory_mode: rolling-1}
  - {stage: 2, name: b, allowed_tools: [], memory_mode: rolling-3}
  - {stage: 3, name: c, allowed_tools: [], memory_mode: rolling-7}
  - {stage: 4, name: d, allowed_tools: [], memory_mode: rolling-21-rag}
  - {stage: 5, name: e, allowed_tools: [], memory_mode: vector-rag-all}
  - {stage: 6, name: f, allowed_tools: [], memory_mode: full-procedural}
`)
	_, err := LoadFromYAML(raw)
	if err == nil {
		t.Fatal("empty memory_mode must error")
	}
}

func TestLoadFromYAML_rejectsMonotonicityBreak(t *testing.T) {
	// Stage 2 drops a tool that Stage 1 had → ADR-149 monotonicity violation.
	raw := []byte(`stages:
  - {stage: 0, name: egg, allowed_tools: [], memory_mode: stateless}
  - {stage: 1, name: baby, allowed_tools: [cite_atom], memory_mode: rolling-1}
  - {stage: 2, name: fledgling, allowed_tools: [atom_search], memory_mode: rolling-3}
  - {stage: 3, name: c, allowed_tools: [cite_atom, atom_search], memory_mode: rolling-7}
  - {stage: 4, name: d, allowed_tools: [cite_atom, atom_search], memory_mode: rolling-21-rag}
  - {stage: 5, name: e, allowed_tools: [cite_atom, atom_search], memory_mode: vector-rag-all}
  - {stage: 6, name: f, allowed_tools: [cite_atom, atom_search], memory_mode: full-procedural}
`)
	_, err := LoadFromYAML(raw)
	if err == nil {
		t.Fatal("monotonicity violation must error; got nil")
	}
	if !strings.Contains(err.Error(), "monotonicity") {
		t.Errorf("error should mention monotonicity; got %v", err)
	}
}

func TestLoadFromYAML_rejectsMissingStage(t *testing.T) {
	// Skips stage 3.
	raw := []byte(`stages:
  - {stage: 0, name: egg, allowed_tools: [], memory_mode: stateless}
  - {stage: 1, name: a, allowed_tools: [], memory_mode: rolling-1}
  - {stage: 2, name: b, allowed_tools: [], memory_mode: rolling-3}
  - {stage: 4, name: d, allowed_tools: [], memory_mode: rolling-21-rag}
  - {stage: 5, name: e, allowed_tools: [], memory_mode: vector-rag-all}
  - {stage: 6, name: f, allowed_tools: [], memory_mode: full-procedural}
  - {stage: 6, name: g, allowed_tools: [], memory_mode: full-procedural}
`)
	_, err := LoadFromYAML(raw)
	if err == nil {
		t.Fatal("missing stage must error")
	}
}

func TestResolve_nilTable(t *testing.T) {
	var t1 *CapabilityTable
	_, err := t1.Resolve(0)
	if err == nil {
		t.Fatal("nil table must error")
	}
}

// ---------- fakes (minimal subset) ----------

type fakeState struct{ m map[string]any }

func (f *fakeState) Get(k string) (any, error) {
	if v, ok := f.m[k]; ok {
		return v, nil
	}
	return nil, session.ErrStateKeyNotExist
}

func (f *fakeState) Set(k string, v any) error {
	f.m[k] = v
	return nil
}

func (f *fakeState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range f.m {
			if !yield(k, v) {
				return
			}
		}
	}
}

type fakeSession struct{ state *fakeState }

func (f *fakeSession) ID() string                { return "fake" }
func (f *fakeSession) AppName() string           { return "fake" }
func (f *fakeSession) UserID() string            { return "fake" }
func (f *fakeSession) State() session.State      { return f.state }
func (f *fakeSession) Events() session.Events    { return nil }
func (f *fakeSession) LastUpdateTime() time.Time { return time.Time{} }

type fakeInvocationCtx struct {
	ctx     context.Context
	session *fakeSession
}

func newFakeInvocationCtx(state *fakeState) *fakeInvocationCtx {
	return &fakeInvocationCtx{
		ctx:     context.Background(),
		session: &fakeSession{state: state},
	}
}

func (f *fakeInvocationCtx) Deadline() (time.Time, bool) { return f.ctx.Deadline() }
func (f *fakeInvocationCtx) Done() <-chan struct{}       { return f.ctx.Done() }
func (f *fakeInvocationCtx) Err() error                  { return f.ctx.Err() }
func (f *fakeInvocationCtx) Value(key any) any           { return f.ctx.Value(key) }

func (f *fakeInvocationCtx) Agent() agent.Agent          { return nil }
func (f *fakeInvocationCtx) Artifacts() agent.Artifacts  { return nil }
func (f *fakeInvocationCtx) Memory() agent.Memory        { return nil }
func (f *fakeInvocationCtx) Session() session.Session    { return f.session }
func (f *fakeInvocationCtx) InvocationID() string        { return "fake-inv" }
func (f *fakeInvocationCtx) Branch() string              { return "" }
func (f *fakeInvocationCtx) UserContent() *genai.Content { return nil }
func (f *fakeInvocationCtx) RunConfig() *agent.RunConfig { return nil }
func (f *fakeInvocationCtx) EndInvocation()              {}
func (f *fakeInvocationCtx) Ended() bool                 { return false }
func (f *fakeInvocationCtx) WithContext(ctx context.Context) agent.InvocationContext {
	return &fakeInvocationCtx{ctx: ctx, session: f.session}
}

func contains(haystack []string, needle string) bool {
	for _, h := range haystack {
		if h == needle {
			return true
		}
	}
	return false
}

// sameStringSet reports whether a and b contain the same elements (order- and
// duplicate-insensitive) — used for exact per-stage innate-kit equality.
func sameStringSet(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	seen := map[string]int{}
	for _, s := range a {
		seen[s]++
	}
	for _, s := range b {
		seen[s]--
	}
	for _, n := range seen {
		if n != 0 {
			return false
		}
	}
	return true
}
