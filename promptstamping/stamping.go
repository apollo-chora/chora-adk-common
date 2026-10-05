// Package promptstamping is the shared ADR-197 M-A primitive: at prompt-compose
// time every Chora ADK Go agent stamps the prompt_version, a deterministic
// content hash of the rendered system prompt, and the structured composition
// conditions onto the active active trace span. This is the explainability
// substrate for IMDA D2 transparency — O+ reconstructs "which prompt version +
// which conditions produced this decision" from these span attributes.
//
// The primitive NEVER alters the rendered prompt. It wraps an existing
// llmagent.InstructionProvider, observes the composed string, and returns it
// unchanged — so behaviour stays byte-identical (ADR-197 behaviour-neutral
// guarantee). It is fail-soft: a nil condition extractor yields empty
// conditions and span stamping is a no-op when no span is recording.
//
// The stamped prompt_version is STATE-DERIVED (ADR-197 M-B): when the
// orchestrator has resolved a version into session state under
// StateKeyResolvedPromptVersion, WithStamping prefers it; otherwise it falls
// back to the agentconfig-embedded version passed at wrap time. This is
// behaviour-neutral — absent the state key, the span carries the same version
// as before, and the rendered prompt is never affected either way.
//
// EVERY ADK Go crew (qgen question + critic, oe evaluator + moderator,
// moderation, familiar) wraps its InstructionProvider with WithStamping so a
// single seam covers all agents.
package promptstamping

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"sort"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/session"
)

// StateKeyResolvedPromptVersion is the session-state key under which the
// orchestrator stamps the resolved prompt version (ADR-197 M-B). When present
// and non-empty, WithStamping stamps this value instead of the agentconfig
// fallback passed at wrap time. Absent ⇒ the fallback (current behaviour).
const StateKeyResolvedPromptVersion = "resolved_prompt_version"

// Evidence is the deterministic prompt-composition evidence captured at compose
// time. The same (PromptVersion, Conditions) inputs always reproduce the same
// ContentHash because composition is a pure function (ADR-141 D2).
type Evidence struct {
	// PromptVersion is the resolved template version (embedded default "vN" in
	// M-A; a registry version once ADR-197 M-B lands).
	PromptVersion string
	// ContentHash is the lowercase hex SHA-256 of the rendered system prompt —
	// a PII-safe, reproducible citation of the exact instruction sent to the LLM.
	ContentHash string
	// Conditions maps each composition discriminant to its resolved value
	// (e.g. intent=new_question, question_type=mcq). Drives the O+ condition→block view.
	Conditions map[string]string
}

// ContentHash returns the lowercase hex SHA-256 of the rendered prompt.
func ContentHash(prompt string) string {
	sum := sha256.Sum256([]byte(prompt))
	return hex.EncodeToString(sum[:])
}

// ConditionExtractor pulls the composition discriminants from readonly session
// state. Implementations MUST be deterministic and side-effect-free; each agent
// supplies one mirroring its own TaskContext discriminants.
type ConditionExtractor func(state session.ReadonlyState) map[string]string

// SpanAttributes renders the Evidence as OTLP span attributes. Condition keys
// are emitted as chora.prompt.condition.<key>, sorted for deterministic output.
func (e Evidence) SpanAttributes() []attribute.KeyValue {
	attrs := []attribute.KeyValue{
		attribute.String("chora.prompt.version", e.PromptVersion),
		attribute.String("chora.prompt.content_hash", e.ContentHash),
	}
	keys := make([]string, 0, len(e.Conditions))
	for k := range e.Conditions {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		attrs = append(attrs, attribute.String("chora.prompt.condition."+k, e.Conditions[k]))
	}
	return attrs
}

// Stamp sets the evidence attributes on the active span in ctx. Safe to call
// when no span is recording (no-op) — mirrors manaplugin.EmitBalanceSpan.
func Stamp(ctx context.Context, e Evidence) {
	span := trace.SpanFromContext(ctx)
	if span == nil || !span.IsRecording() {
		return
	}
	span.SetAttributes(e.SpanAttributes()...)
}

// resolvedVersion prefers the orchestrator-resolved prompt version from session
// state (StateKeyResolvedPromptVersion) when present and non-empty; otherwise it
// returns the agentconfig fallback. Fail-soft: a nil state, a missing key, or a
// non-string/empty value all yield the fallback — keeping the span version
// behaviour-neutral when no override has been resolved.
func resolvedVersion(state session.ReadonlyState, fallback string) string {
	if state == nil {
		return fallback
	}
	v, err := state.Get(StateKeyResolvedPromptVersion)
	if err != nil {
		return fallback
	}
	if s, ok := v.(string); ok && s != "" {
		return s
	}
	return fallback
}

// WithStamping wraps an llmagent.InstructionProvider. On each turn it calls
// inner to render the prompt, computes the content hash, resolves the prompt
// version (state StateKeyResolvedPromptVersion when present, else the passed
// fallback), extracts the condition set, stamps the evidence on the active span,
// and returns the prompt UNCHANGED. The version argument is the FALLBACK default
// (the agentconfig-embedded version) — state, when set, wins. A nil extractor
// yields empty conditions. Inner errors propagate verbatim and nothing is
// stamped (a failed compose has no evidence).
func WithStamping(version string, extract ConditionExtractor, inner llmagent.InstructionProvider) llmagent.InstructionProvider {
	return func(ctx agent.ReadonlyContext) (string, error) {
		prompt, err := inner(ctx)
		if err != nil {
			return "", err
		}
		state := ctx.ReadonlyState()
		ev := Evidence{PromptVersion: resolvedVersion(state, version), ContentHash: ContentHash(prompt)}
		if extract != nil {
			ev.Conditions = extract(state)
		}
		Stamp(ctx, ev)
		return prompt, nil
	}
}
