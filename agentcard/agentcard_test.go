package agentcard

import (
	"encoding/json"
	"strings"
	"testing"
)

// ---------- Builder ----------

func TestBuilder_minimalCardBuilds(t *testing.T) {
	card, err := NewBuilder().
		WithName("familiar_companion").
		WithVersion("1.0.0").
		WithAGID("agid:familiar:01957c8c-aaaa-7000-bbbb-cccccccccccc").
		Build()
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if card.Name != "familiar_companion" {
		t.Errorf("want name; got %q", card.Name)
	}
	if card.Version != "1.0.0" {
		t.Errorf("want version; got %q", card.Version)
	}
	if card.AGID == "" {
		t.Error("AGID must be set")
	}
}

func TestBuilder_requiresName(t *testing.T) {
	_, err := NewBuilder().
		WithVersion("1.0.0").
		WithAGID("agid:x").
		Build()
	if err == nil {
		t.Fatal("want error when Name missing; got nil")
	}
	if !strings.Contains(err.Error(), "name") {
		t.Errorf("error should mention name; got %v", err)
	}
}

func TestBuilder_requiresVersion(t *testing.T) {
	_, err := NewBuilder().
		WithName("x").
		WithAGID("agid:x").
		Build()
	if err == nil {
		t.Fatal("want error when Version missing; got nil")
	}
}

func TestBuilder_requiresAGID(t *testing.T) {
	_, err := NewBuilder().
		WithName("x").
		WithVersion("1.0.0").
		Build()
	if err == nil {
		t.Fatal("want error when AGID missing; got nil")
	}
}

func TestBuilder_appendsCapabilities(t *testing.T) {
	card, _ := NewBuilder().
		WithName("familiar").
		WithVersion("1.0.0").
		WithAGID("agid:x").
		WithCapability(Capability{Name: "study_conversation", Description: "Multi-turn tutoring"}).
		WithCapability(Capability{Name: "persona_voice", Description: "Set persona voice"}).
		Build()
	if len(card.Capabilities) != 2 {
		t.Errorf("want 2 capabilities; got %d", len(card.Capabilities))
	}
	if card.Capabilities[0].Name != "study_conversation" {
		t.Errorf("want first capability study_conversation; got %q", card.Capabilities[0].Name)
	}
}

func TestBuilder_appendsEndpoints(t *testing.T) {
	card, _ := NewBuilder().
		WithName("familiar").
		WithVersion("1.0.0").
		WithAGID("agid:x").
		WithEndpoint(Endpoint{Type: "rest", BaseURL: "https://a2a.chora.site/agents/familiar"}).
		Build()
	if len(card.Endpoints) != 1 {
		t.Fatalf("want 1 endpoint; got %d", len(card.Endpoints))
	}
	if card.Endpoints[0].BaseURL != "https://a2a.chora.site/agents/familiar" {
		t.Errorf("want base URL; got %q", card.Endpoints[0].BaseURL)
	}
}

func TestBuilder_appendsComplianceTags(t *testing.T) {
	card, _ := NewBuilder().
		WithName("familiar").
		WithVersion("1.0.0").
		WithAGID("agid:x").
		WithComplianceTag("imda-d1-accountability").
		WithComplianceTag("imda-d2-transparency").
		WithComplianceTag("imda-d3-safety_and_robustness").
		Build()
	if len(card.ComplianceTags) != 3 {
		t.Errorf("want 3 compliance tags; got %d", len(card.ComplianceTags))
	}
}

func TestBuilder_setsOwnerTeamAndDescription(t *testing.T) {
	card, _ := NewBuilder().
		WithName("familiar").
		WithVersion("1.0.0").
		WithAGID("agid:x").
		WithOwnerTeam("Team 1 (Content)").
		WithDescription("Per-learner RPG companion").
		Build()
	if card.OwnerTeam != "Team 1 (Content)" {
		t.Errorf("want owner team; got %q", card.OwnerTeam)
	}
	if card.Description != "Per-learner RPG companion" {
		t.Errorf("want description; got %q", card.Description)
	}
}

func TestBuilder_setsSchemaVersionDefault(t *testing.T) {
	card, _ := NewBuilder().
		WithName("x").
		WithVersion("1.0.0").
		WithAGID("agid:x").
		Build()
	if card.SchemaVersion != A2ASchemaVersion {
		t.Errorf("schemaVersion should default to A2A v1.0; got %q", card.SchemaVersion)
	}
}

func TestBuilder_immutableAfterBuild(t *testing.T) {
	// Mutating slices on the original Builder after Build() must not
	// retroactively change the Built card.
	b := NewBuilder().
		WithName("familiar").
		WithVersion("1.0.0").
		WithAGID("agid:x").
		WithCapability(Capability{Name: "a"})

	card1, _ := b.Build()
	b = b.WithCapability(Capability{Name: "b"})
	card2, _ := b.Build()

	if len(card1.Capabilities) != 1 {
		t.Errorf("card1 should be unaffected by subsequent builder mutation; got %d", len(card1.Capabilities))
	}
	if len(card2.Capabilities) != 2 {
		t.Errorf("card2 should reflect the new capability; got %d", len(card2.Capabilities))
	}
}

// ---------- JSON serialisation ----------

func TestCard_MarshalJSON_canonicalFields(t *testing.T) {
	card, _ := NewBuilder().
		WithName("familiar_companion").
		WithVersion("1.0.0").
		WithAGID("agid:familiar:abc").
		WithDescription("Per-learner RPG companion").
		WithOwnerTeam("Team 1").
		WithCapability(Capability{Name: "study_conversation"}).
		WithEndpoint(Endpoint{Type: "rest", BaseURL: "https://a2a.chora.site/agents/familiar"}).
		WithComplianceTag("imda-d1-accountability").
		Build()

	bs, err := json.Marshal(card)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	str := string(bs)

	// Canonical A2A v1.0 wire field names (camelCase).
	for _, want := range []string{
		`"schemaVersion":"v1.0"`,
		`"name":"familiar_companion"`,
		`"version":"1.0.0"`,
		`"agid":"agid:familiar:abc"`,
		`"description":"Per-learner RPG companion"`,
		`"ownerTeam":"Team 1"`,
		`"capabilities":`,
		`"endpoints":`,
		`"complianceTags":`,
	} {
		if !strings.Contains(str, want) {
			t.Errorf("JSON should contain %q; got %s", want, str)
		}
	}
}

func TestCard_MarshalJSON_omitsEmptyFields(t *testing.T) {
	card, _ := NewBuilder().
		WithName("x").
		WithVersion("1.0.0").
		WithAGID("agid:x").
		Build()

	bs, _ := json.Marshal(card)
	str := string(bs)
	if strings.Contains(str, `"description":""`) {
		t.Error("empty description should be omitted from JSON")
	}
	if strings.Contains(str, `"complianceTags":null`) || strings.Contains(str, `"complianceTags":[]`) {
		// Either omission OR empty array is acceptable; null is not.
		if strings.Contains(str, `"complianceTags":null`) {
			t.Error("complianceTags should be omitted, not null")
		}
	}
}

func TestCard_UnmarshalJSON_roundTrips(t *testing.T) {
	original, _ := NewBuilder().
		WithName("familiar").
		WithVersion("1.0.0").
		WithAGID("agid:familiar:abc").
		WithCapability(Capability{Name: "study_conversation"}).
		WithEndpoint(Endpoint{Type: "rest", BaseURL: "https://a2a.chora.site/agents/familiar"}).
		WithComplianceTag("imda-d1-accountability").
		Build()

	bs, _ := json.Marshal(original)

	var roundTripped Card
	if err := json.Unmarshal(bs, &roundTripped); err != nil {
		t.Fatalf("unexpected: %v", err)
	}

	if roundTripped.Name != original.Name {
		t.Errorf("name lost in roundtrip; want %q got %q", original.Name, roundTripped.Name)
	}
	if len(roundTripped.Capabilities) != 1 {
		t.Errorf("capabilities lost in roundtrip; got %d", len(roundTripped.Capabilities))
	}
	if len(roundTripped.ComplianceTags) != 1 {
		t.Errorf("compliance tags lost in roundtrip; got %d", len(roundTripped.ComplianceTags))
	}
}
