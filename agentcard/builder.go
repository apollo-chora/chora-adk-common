package agentcard

import (
	"errors"
)

// Builder constructs a Card via fluent With* methods. Build() validates
// the required fields (Name + Version + AGID) and returns the immutable
// Card by value.
//
// All With* methods return a NEW Builder (copy-on-write). The same
// Builder can be Built multiple times and each result is independent of
// subsequent mutations.
type Builder struct {
	card Card
}

// NewBuilder returns an empty Builder.
func NewBuilder() *Builder {
	return &Builder{card: Card{SchemaVersion: A2ASchemaVersion}}
}

// WithName sets Card.Name.
func (b *Builder) WithName(name string) *Builder {
	out := b.copy()
	out.card.Name = name
	return out
}

// WithVersion sets Card.Version (semver).
func (b *Builder) WithVersion(version string) *Builder {
	out := b.copy()
	out.card.Version = version
	return out
}

// WithDescription sets Card.Description.
func (b *Builder) WithDescription(description string) *Builder {
	out := b.copy()
	out.card.Description = description
	return out
}

// WithAGID sets Card.AGID. Required per ADR-132 — agents have stable
// identities distinct from GCID.
func (b *Builder) WithAGID(agid string) *Builder {
	out := b.copy()
	out.card.AGID = agid
	return out
}

// WithOwnerTeam sets Card.OwnerTeam.
func (b *Builder) WithOwnerTeam(team string) *Builder {
	out := b.copy()
	out.card.OwnerTeam = team
	return out
}

// WithCapability appends a Capability.
func (b *Builder) WithCapability(c Capability) *Builder {
	out := b.copy()
	out.card.Capabilities = append(out.card.Capabilities, c)
	return out
}

// WithEndpoint appends an Endpoint.
func (b *Builder) WithEndpoint(e Endpoint) *Builder {
	out := b.copy()
	out.card.Endpoints = append(out.card.Endpoints, e)
	return out
}

// WithComplianceTag appends a compliance tag. Use canonical IMDA labels
// per ADR-141 (e.g., "imda-d1-accountability", "imda-d2-transparency",
// "imda-d3-safety_and_robustness", "imda-d4-fairness_and_human_oversight").
func (b *Builder) WithComplianceTag(tag string) *Builder {
	out := b.copy()
	out.card.ComplianceTags = append(out.card.ComplianceTags, tag)
	return out
}

// Build validates the required fields and returns the immutable Card.
// Required: Name, Version, AGID.
func (b *Builder) Build() (Card, error) {
	if b.card.Name == "" {
		return Card{}, errors.New("agentcard: name required (per A2A v1.0)")
	}
	if b.card.Version == "" {
		return Card{}, errors.New("agentcard: version required (semver)")
	}
	if b.card.AGID == "" {
		return Card{}, errors.New("agentcard: AGID required (per ADR-132)")
	}
	// Defensive copy of slices so post-Build mutations on the Builder
	// don't bleed into the returned Card.
	out := b.card
	out.Capabilities = append([]Capability(nil), b.card.Capabilities...)
	out.Endpoints = append([]Endpoint(nil), b.card.Endpoints...)
	out.ComplianceTags = append([]string(nil), b.card.ComplianceTags...)
	return out, nil
}

// copy returns a deep enough copy that subsequent With* calls on either
// the parent or the child Builder don't interfere with each other.
func (b *Builder) copy() *Builder {
	out := &Builder{card: b.card}
	out.card.Capabilities = append([]Capability(nil), b.card.Capabilities...)
	out.card.Endpoints = append([]Endpoint(nil), b.card.Endpoints...)
	out.card.ComplianceTags = append([]string(nil), b.card.ComplianceTags...)
	return out
}
