// Package agentcard implements Agent Card publishing per A2A v1.0 +
// ADR-132 (A2A protocol integration).
//
// Per crew-composition skill: every crew (P1-P8) publishes an Agent
// Card describing its identity, capabilities, owner team, AGID, and
// compliance posture. The card is served by every crew launcher at
// the canonical A2A well-known path (`/.well-known/agent-card`).
//
// AGID is the agent's stable identity (distinct from GCID per ADR-132
// + ddd-enforcement rule #10 — "agents CANNOT hold TenantMembership").
//
// Today crews own their own `agent_card.yaml`. This package provides
// the Go-side authoring + serving primitives so multiple crews can
// share the canonical wire format + the HTTP handler.
//
// Schema: A2A v1.0 (https://google.github.io/A2A/). Field names are
// camelCase on the wire — the Go struct tags handle the conversion.
package agentcard

// A2ASchemaVersion is the wire schema version stamped on every card.
// Bump when the A2A protocol upgrades.
const A2ASchemaVersion = "v1.0"

// A2AWellKnownPath is the canonical URI fragment where Agent Cards are
// served per A2A v1.0. Crew launchers MUST register their card handler
// at this path on their public HTTP surface (typically a2a.chora.site
// per ingress topology in chora_topology memory).
const A2AWellKnownPath = "/.well-known/agent-card"

// Card is the A2A v1.0 Agent Card. Wire shape uses camelCase JSON; Go
// struct uses PascalCase. Pre-build mutability is gated behind Builder
// — once Build() returns a Card, treat it as immutable (passed by value).
type Card struct {
	// SchemaVersion is the A2A schema version (always "v1.0" today).
	SchemaVersion string `json:"schemaVersion"`

	// Name is the agent identifier (snake_case; e.g., "familiar_companion").
	// Maps 1:1 to the crew kind for crews that deploy as a single agent.
	Name string `json:"name"`

	// Version is the semver of THIS agent (not the schema).
	Version string `json:"version"`

	// Description is human-readable agent purpose.
	Description string `json:"description,omitempty"`

	// AGID is the agent's stable identity per ADR-132. Distinct from
	// GCID — agents CANNOT hold TenantMembership.
	AGID string `json:"agid"`

	// OwnerTeam is the team name (e.g., "Team 1 (Content)") per the
	// 3-team structure in project_chora_topology memory.
	OwnerTeam string `json:"ownerTeam,omitempty"`

	// Capabilities enumerates the skills / tools this agent exposes
	// to external callers per A2A protocol.
	Capabilities []Capability `json:"capabilities,omitempty"`

	// Endpoints is the list of transport endpoints (REST, gRPC, etc.)
	// where this agent can be invoked.
	Endpoints []Endpoint `json:"endpoints,omitempty"`

	// ComplianceTags carries IMDA dimension labels + other compliance
	// posture markers (per ADR-141 canonical labels).
	ComplianceTags []string `json:"complianceTags,omitempty"`
}

// Capability is one A2A v1.0 capability entry.
type Capability struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`

	// InputSchema is a JSON Schema reference (e.g., "$ref:study_conversation_input.json").
	InputSchema string `json:"inputSchema,omitempty"`

	// OutputSchema is the response JSON Schema reference.
	OutputSchema string `json:"outputSchema,omitempty"`

	// ConsentScopeRequired declares the FamiliarExposureGrant scope
	// external callers MUST hold to invoke this capability (per ADR-132).
	ConsentScopeRequired string `json:"consentScopeRequired,omitempty"`
}

// Endpoint is one A2A v1.0 endpoint entry.
type Endpoint struct {
	Type    string `json:"type"`              // "rest" | "grpc" | "websocket"
	BaseURL string `json:"baseUrl,omitempty"` // base URL (auth handled separately)
}
