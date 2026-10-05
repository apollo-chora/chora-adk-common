// Package modelgatewayclient adapts the chora-model-gateway gRPC Invoke
// surface (chora-contracts/proto/services/model_gateway_service.proto) to
// the ADK Go `model.LLM` interface (google.golang.org/adk/model).
//
// Phase 3.1 cutover (ADR-163): replaces per-agent `gemini.NewModel` calls
// with a single chokepoint through the gateway. The gateway composes
// per-agent routing + Model Armor PRE/POST + per-tenant budget +
// token-usage ledger emission inside the request path.
//
// Full mana umbrella (ADR-177): this is the SHARED tool-aware client (one copy
// in chora-adk-common, imported by every crew — the per-agent qgen/oe copies
// were retired). Beyond the legacy flat text path it carries the structured
// genai conversation + tool declarations through Invoke so TOOL-CALLING crews
// (familiar / recommender / moderation) route every LLM turn through the one
// metered gateway instead of calling Gemini directly:
//   - sends `contents_json` (genai []*Content) + `tools_json` (genai []*Tool);
//   - parses `tool_calls_json` back into genai FunctionCall parts so the ADK
//     runtime drives the tool loop locally (the gateway is single-turn — it
//     cannot run agent-side Go functiontool handlers);
//   - sets `action_code` ONLY on the turn-initiating Invoke (per-turn metering,
//     ADR-177 §5) so one user-facing turn costs one debit, not one-per-LLM-call.
//
// Wire shape (caller → gateway):
//
//	ADK llmagent.GenerateContent
//	  └─ adkmodel.LLM.GenerateContent
//	       └─ modelgatewayclient.client.GenerateContent
//	            └─ static bearer token (CHORA_GATEWAY_TOKEN env) or
//	               plaintext for local dev (Insecure)
//	            └─ gRPC Invoke over TLS to the gateway endpoint
//	            └─ chora-model-gateway domain.Service.Invoke
//	                 └─ vendor dispatch (Gemini / Gemma / BYOA)
//
// Tenancy + identity:
//   - `agent_id` is set per binary (qgen_question, qgen_critic, ...).
//   - `tenant_id` + `gcid` are propagated via CHORA_GATEWAY_TENANT_ID /
//     CHORA_GATEWAY_GCID env vars set by adkgo deploy. Phase 3.1c will
//     move these to ADK session.state when the orchestrator propagation
//     contract is finalised.
//
// Streaming:
//   - The proto Invoke is unary (single request, single response). Phase 6
//     of ADR-163 introduces streaming. For now we yield exactly one
//     LLMResponse from the iterator regardless of caller's `stream` arg.
package modelgatewayclient

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"os"
	"strings"
	"time"

	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"go.opentelemetry.io/otel/trace"
	"golang.org/x/oauth2"
	"google.golang.org/genai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/credentials/oauth"
	"google.golang.org/protobuf/types/known/structpb"

	mgv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/services/model_gateway/v1"

	adkmodel "google.golang.org/adk/model"
)

// Config carries everything the gateway client needs to dial + identify
// itself. Zero-value Config is invalid (New returns an error).
type Config struct {
	// Endpoint — gRPC target, e.g. "gateway.chora.site:443". REQUIRED.
	Endpoint string

	// LogicalModelID — caller-supplied model name passed verbatim as
	// InvokeRequest.logical_model_id. The gateway resolves through its
	// per-agent routing policy YAML and MAY substitute (e.g. Pro→Flash
	// during budget pressure). REQUIRED.
	LogicalModelID string

	// FallbackModelIDs — agent-declared ordered fallback chain passed as
	// InvokeRequest.fallback_logical_model_ids. The gateway HONOURS this
	// chain (tries each in order on primary dispatch failure) rather than
	// hardcoding a per-tier ladder — AGENT-DRIVEN tiering (CR qgen
	// 2026-06-01). Sourced from the sub-agent's build-time YAML config.
	// OPTIONAL — empty = single-shot.
	FallbackModelIDs []string

	// AgentID — calling-agent identifier for the gateway's routing policy
	// lookup. Matches the binary's logical name in the agent registry
	// (e.g., "qgen_question").
	// REQUIRED.
	AgentID string

	// CrewKind — optional crew tag for cost-attribution dashboards
	// (e.g., "qgen"). Optional but recommended.
	CrewKind string

	// Surface is the crew id of the invoking agent, stamped as
	// InvokeRequest.surface on EVERY Invoke, tool-result continuations
	// included (ADR-254 D7 / ADR-252 Q1): one of companion_chat,
	// companion_diagnosis, kg_exploration, qgen, oe_grading,
	// content_recommender, content_moderation, duel_atom_smith,
	// profile_conjurer. The gateway refuses an ABSENT surface with
	// FAILED_PRECONDITION "surface_unstamped" before any debit, so this is
	// REQUIRED here rather than defaulted: a binary that forgot it must fail
	// at boot, not at its first model call.
	Surface string

	// ActionCode — the crew's BASE mana action_code for umbrella metering
	// (ADR-177 §5 + ADR-142 §4). The client sets InvokeRequest.action_code
	// to this value ONLY on a turn-initiating Invoke (the request whose last
	// content is a `user` message); tool-result continuations send an empty
	// action_code so the gateway debits ONCE per user-facing turn, not once
	// per tool-loop iteration. A per-request override may be stamped into
	// session state (e.g. familiar's per-tier `familiar_chat_turn_{tier}`) by
	// NewTenantPropagationPlugin's ActionCodeResolver and takes precedence.
	//
	// OPTIONAL — empty disables gateway-side metering for this crew (the
	// legacy un-metered behaviour; used while a crew still carries an ad-hoc
	// debit on the calling service, to avoid double-metering — see ADR-177 §6).
	ActionCode string

	// TenantID — tenant scope for RLS + ledger attribution. REQUIRED.
	// Phase 3.1: sourced from CHORA_GATEWAY_TENANT_ID env var.
	TenantID string

	// GCID — actor identity (learner GCID or agent AGID). REQUIRED.
	// Phase 3.1: sourced from CHORA_GATEWAY_GCID env var.
	GCID string

	// Audience — the audience the gateway token was minted for. Informational
	// on the cloud-neutral path (the static bearer token is sent as-is); kept
	// for deployments that mint their own audience-bound tokens.
	Audience string

	// CallTimeout — per-Invoke deadline. Defaults to 90s (some multimodal
	// LLM calls can legitimately take 60-90 s). HTTPRoute upstream is
	// 120 s; keep this under that for clean client-side timeout vs server
	// abort distinction.
	CallTimeout time.Duration

	// Insecure — when true, dials the gateway with plaintext gRPC
	// (insecure.NewCredentials) instead of TLS + bearer token. For LOCAL
	// DEV ONLY: set via CHORA_GATEWAY_INSECURE=1 env var when pointing
	// agents at a local gateway (the compose stack). NEVER set in prod —
	// the client MUST use TLS + CHORA_GATEWAY_TOKEN in prod.
	Insecure bool

	// dialOptsExtra — test seam for unit tests with bufconn / fake server.
	// Not part of the public surface.
	dialOptsExtra []grpc.DialOption
}

// New constructs a gateway-fronted adkmodel.LLM. Fails LOUD on missing
// required config per feedback_no_stubs_real_wiring (no silent in-memory
// fallback).
func New(ctx context.Context, cfg Config) (adkmodel.LLM, error) {
	if cfg.Endpoint == "" {
		return nil, errors.New("modelgatewayclient: Endpoint required")
	}
	if cfg.LogicalModelID == "" {
		return nil, errors.New("modelgatewayclient: LogicalModelID required")
	}
	if cfg.AgentID == "" {
		return nil, errors.New("modelgatewayclient: AgentID required")
	}
	if cfg.Surface == "" {
		return nil, errors.New("modelgatewayclient: Surface required (the crew id, ADR-254 D7)")
	}
	if cfg.TenantID == "" {
		return nil, errors.New("modelgatewayclient: TenantID required (CHORA_GATEWAY_TENANT_ID)")
	}
	if cfg.GCID == "" {
		return nil, errors.New("modelgatewayclient: GCID required (CHORA_GATEWAY_GCID)")
	}
	if cfg.Audience == "" {
		cfg.Audience = "https://gateway.chora.site"
	}
	if cfg.CallTimeout == 0 {
		cfg.CallTimeout = 90 * time.Second
	}

	conn, err := dialGateway(ctx, cfg)
	if err != nil {
		return nil, err
	}

	return &client{
		cfg:  cfg,
		conn: conn,
		grpc: mgv1.NewModelGatewayServiceClient(conn),
	}, nil
}

// dialGateway opens the gRPC connection every gateway client (the ADK LLM
// wrapper and the image invoker) shares: TLS + a static bearer token from
// CHORA_GATEWAY_TOKEN in production, plaintext for local dev, the caller's
// own options in tests.
func dialGateway(ctx context.Context, cfg Config) (*grpc.ClientConn, error) {
	// Production path adds TLS + bearer-token PerRPC credentials. Test mode
	// (dialOptsExtra non-empty) supplies its own credentials + dialer and
	dialOpts := []grpc.DialOption{
		grpc.WithStatsHandler(otelgrpc.NewClientHandler()),
	}
	if cfg.Insecure {
		// LOCAL DEV ONLY — plaintext gRPC to a local gateway (no TLS,
		// no token). Set via CHORA_GATEWAY_INSECURE=1 env var.
		dialOpts = append(dialOpts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	} else if len(cfg.dialOptsExtra) == 0 {
		token := strings.TrimSpace(os.Getenv(EnvGatewayToken))
		if token == "" {
			return nil, fmt.Errorf("modelgatewayclient: %s is required for authenticated "+
				"gateway calls (or Config.Insecure for local dev)", EnvGatewayToken)
		}
		dialOpts = append(dialOpts,
			grpc.WithTransportCredentials(credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12})),
			grpc.WithPerRPCCredentials(oauth.TokenSource{TokenSource: staticTokenSource{token: token}}),
		)
	} else {
		dialOpts = append(dialOpts, cfg.dialOptsExtra...)
	}

	conn, err := grpc.NewClient(cfg.Endpoint, dialOpts...)
	if err != nil {
		return nil, fmt.Errorf("modelgatewayclient: grpc.NewClient(%s): %w", cfg.Endpoint, err)
	}
	return conn, nil
}

// EnvGatewayToken names the env var carrying the static bearer token the
// client sends as `Authorization: Bearer <token>` on every Invoke. The local
// model gateway does not enforce token auth; deployments that front it with
// an authenticating proxy read the token from the environment (never inline,
// per feedback_no_inline_config).
const EnvGatewayToken = "CHORA_GATEWAY_TOKEN"

// staticTokenSource is an oauth2.TokenSource over a fixed bearer token.
type staticTokenSource struct{ token string }

func (s staticTokenSource) Token() (*oauth2.Token, error) {
	return &oauth2.Token{AccessToken: s.token, TokenType: "Bearer"}, nil
}

// client is the adkmodel.LLM implementation backed by chora-model-gateway.
type client struct {
	cfg  Config
	conn *grpc.ClientConn
	grpc mgv1.ModelGatewayServiceClient
}

// Name — implements adkmodel.LLM. Returns the logical model ID; the actual
// served model is reported back by the gateway in InvokeResponse.model_version.
func (c *client) Name() string { return c.cfg.LogicalModelID }

// GenerateContent — implements adkmodel.LLM. Wraps the unary Invoke RPC as
// a single-item iterator (stream=true callers receive the same single
// terminal response).
func (c *client) GenerateContent(ctx context.Context, req *adkmodel.LLMRequest, _ bool) iter.Seq2[*adkmodel.LLMResponse, error] {
	return func(yield func(*adkmodel.LLMResponse, error) bool) {
		if req == nil {
			yield(nil, errors.New("modelgatewayclient: nil LLMRequest"))
			return
		}

		prompt, sysPrompt := flattenContents(req.Contents)

		// ADK's WithInstruction(...) / per-turn InstructionProvider routes the
		// instruction text via req.Config.SystemInstruction (genai.Content),
		// NOT via req.Contents with role="system". The ADK gemini adapter
		// passes Config directly to Vertex Gemini which honours
		// SystemInstruction; we must do the same here or the model receives
		// only the placeholder "Handle the requests as specified in the
		// System Instruction." that ADK's maybeAppendUserContent inserts
		// when Contents is empty, and replies with a generic
		// "I'm not sure what the situation is about". The per-turn
		// instruction takes precedence over any system role part flattened
		// out of Contents; sysPrompt from Contents is appended after when
		// both are present.
		if cfgSys := extractSystemInstruction(req.Config); cfgSys != "" {
			if sysPrompt == "" {
				sysPrompt = cfgSys
			} else {
				sysPrompt = cfgSys + "\n\n" + sysPrompt
			}
		}

		// ADR-177 tool-aware path. Carry the structured genai conversation +
		// tool declarations to the gateway when the turn is genuinely
		// multi-turn or tool-bearing; the gateway's gemini adapter builds the
		// request from `contents_json` (preferring it over the flat `prompt`)
		// and passes `tools_json` through verbatim. Simple single-turn,
		// tool-less text calls (qgen text, image-gen) stay on the proven
		// flat-prompt path (contentsJSON empty) — byte-identical to pre-ADR-177.
		// The flat `prompt`/`system_prompt` are ALWAYS sent: Model Armor
		// PRE screens `prompt`, and they keep back-compat for text-only callers.
		contentsJSON, err := marshalContents(req.Contents)
		if err != nil {
			yield(nil, fmt.Errorf("modelgatewayclient: marshal contents: %w", err))
			return
		}
		toolsJSON, err := marshalTools(req.Config)
		if err != nil {
			yield(nil, fmt.Errorf("modelgatewayclient: marshal tools: %w", err))
			return
		}

		if prompt == "" && contentsJSON == "" {
			yield(nil, errors.New("modelgatewayclient: no user content in LLMRequest"))
			return
		}

		genCfg, err := encodeGenerationConfig(req.Config)
		if err != nil {
			yield(nil, fmt.Errorf("modelgatewayclient: encode generation_config: %w", err))
			return
		}

		// Per-turn metering (ADR-177 §5). Resolve the effective action_code —
		// a per-request override stamped into Config.Labels by the propagation
		// plugin's ActionCodeResolver (e.g. familiar's per-tier code) wins over
		// the crew's static cfg.ActionCode. Set it on the wire ONLY for a
		// turn-initiating Invoke; tool-result continuations send empty so the
		// gateway debits once per user-facing turn, not once per LLM call.
		actionCode := c.cfg.ActionCode
		if lbl := actionCodeFromLabels(req); lbl != "" {
			actionCode = lbl
		}
		if !isTurnInitiating(req.Contents) {
			actionCode = ""
		}

		// Use the caller's model when set (per-call override of cfg default),
		// else fall back to the configured logical model ID.
		logicalModel := req.Model
		if logicalModel == "" {
			logicalModel = c.cfg.LogicalModelID
		}

		// ADR-169 — per-request tenant attribution. The BeforeModelCallback
		// (NewTenantPropagationPlugin) stamps the REQUESTING tenant/gcid (from
		// session state) into req.Config.Labels; honour them so the gateway
		// scopes RLS + ledger + budget to that tenant. Fall back to the
		// env-configured defaults (c.cfg) when a label is absent.
		tenantID, gcid := c.cfg.TenantID, c.cfg.GCID
		if lblTenant, lblGCID := tenantFromLabels(req); lblTenant != "" || lblGCID != "" {
			if lblTenant != "" {
				tenantID = lblTenant
			}
			if lblGCID != "" {
				gcid = lblGCID
			}
		}

		invokeReq := &mgv1.InvokeRequest{
			TenantId:                tenantID,
			Gcid:                    gcid,
			AgentId:                 c.cfg.AgentID,
			CrewKind:                c.cfg.CrewKind,
			LogicalModelId:          logicalModel,
			FallbackLogicalModelIds: c.cfg.FallbackModelIDs,
			Prompt:                  prompt,
			SystemPrompt:            sysPrompt,
			GenerationConfig:        genCfg,
			ContentsJson:            contentsJSON,
			ToolsJson:               toolsJSON,
			ActionCode:              actionCode,
			// ADR-254 D7: the crew id on every Invoke (the gateway's
			// suspension read and the surface gate key on it), and the
			// dispatch idempotency key (R22) the tenant-propagation plugin
			// copied from session state, so a redelivered dispatch is a
			// keyed debit claim at the gateway and bills once. Empty on a
			// non-dispatched call, which the gateway treats as no claim.
			Surface:                c.cfg.Surface,
			DispatchIdempotencyKey: dispatchKeyFromLabels(req),
		}
		// W3C trace context propagation — source from the ACTIVE OTel span
		// (ADK instruments the agent invocation), NOT inbound gRPC metadata.
		// An ADK agent making an LLM call has no inbound gRPC metadata, so the
		// previous metadata.FromIncomingContext read always produced empty
		// values and left the gateway's Invoke span an uncorrelated root. The
		// otelgrpc client handler already injects the same context into gRPC
		// metadata; setting the explicit proto field keeps that channel correct
		// too.
		if sc := trace.SpanContextFromContext(ctx); sc.IsValid() {
			invokeReq.Traceparent = fmt.Sprintf("00-%s-%s-%02x", sc.TraceID(), sc.SpanID(), byte(sc.TraceFlags()))
			if tsv := sc.TraceState().String(); tsv != "" {
				invokeReq.Tracestate = tsv
			}
		}

		callCtx, cancel := context.WithTimeout(ctx, c.cfg.CallTimeout)
		defer cancel()

		resp, err := c.grpc.Invoke(callCtx, invokeReq)
		if err != nil {
			yield(nil, fmt.Errorf("modelgatewayclient: Invoke: %w", err))
			return
		}

		// C3 per-hop token surfacing. The gateway returns provider-reported
		// token counts on InvokeResponse.usage (TokenUsage{input_tokens,
		// output_tokens, cached_tokens, cost_micros}). We surface that usage
		// THREE ways so it reaches every downstream consumer:
		//
		//   1. completion JSON merge — the Python executor's _map_response
		//      reads `tokens_consumed_total` / `input_tokens` / `output_tokens`
		//      off the TERMINAL CANDIDATE JSON (the LLM completion text), which
		//      flows into the orchestrator's pipeline_trace rows + the W6
		//      token-compounding eval. So we merge the three fields into the
		//      completion's top-level JSON object (fail-soft: a non-JSON
		//      completion passes through verbatim — token injection must never
		//      corrupt a candidate).
		//   2. LLMResponse.UsageMetadata (genai shape) — in-process visibility
		//      for any ADK plugin / OTLP span attribute that reads it.
		//   3. CustomMetadata — structpb-serialisable scalars for
		//      ai-cost-tracking dashboards + trace backend.
		//
		// Before C3 the client dropped resp.GetUsage() entirely, so the
		// terminal JSON carried tokens_consumed_total=0 and the per-hop split
		// was unavailable.
		usage := resp.GetUsage()
		completion := mergeTokensIntoCompletion(resp.GetCompletion(), usage)

		custom := map[string]any{
			"chora.gateway.invocation_id":   resp.GetInvocationId(),
			"chora.gateway.vendor":          resp.GetVendor(),
			"chora.gateway.fallback_chain":  stringSliceToAny(resp.GetFallbackChain()),
			"chora.gateway.latency_ms":      resp.GetLatencyMs(),
			"chora.gateway.gateway_version": resp.GetGatewayVersion(),
		}
		var usageMeta *genai.GenerateContentResponseUsageMetadata
		if usage != nil {
			in, out := usage.GetInputTokens(), usage.GetOutputTokens()
			custom["chora.gateway.input_tokens"] = in
			custom["chora.gateway.output_tokens"] = out
			custom["chora.gateway.tokens_consumed_total"] = in + out
			custom["chora.gateway.cost_micros"] = usage.GetCostMicros()
			usageMeta = &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:        int32(in),
				CandidatesTokenCount:    int32(out),
				TotalTokenCount:         int32(in + out),
				CachedContentTokenCount: int32(usage.GetCachedTokens()),
			}
		}

		// ADR-177 tool-calling — when the gateway surfaces model-emitted function
		// calls, build them into the response Content as genai FunctionCall parts
		// so the ADK runtime executes the local functiontool handlers and
		// continues the loop (the gateway is single-turn; the AGENT owns the
		// loop). Falls back to a single text part for a terminal text turn.
		parts, err := buildResponseParts(completion, resp.GetToolCallsJson())
		if err != nil {
			yield(nil, fmt.Errorf("modelgatewayclient: parse tool_calls: %w", err))
			return
		}

		llmResp := &adkmodel.LLMResponse{
			Content: &genai.Content{
				Role:  "model",
				Parts: parts,
			},
			ModelVersion:   resp.GetModelVersion(),
			FinishReason:   mapFinishDetail(resp.GetFinishDetail()),
			UsageMetadata:  usageMeta,
			CustomMetadata: custom,
			TurnComplete:   true,
		}
		yield(llmResp, nil)
	}
}

// Close releases the gRPC connection. Safe to call multiple times.
func (c *client) Close() error {
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// marshalContents serialises the genai []*Content conversation into the
// `contents_json` field (ADR-177). Returns "" (and the flat-prompt path is
// used) for a simple single-turn, tool-less, text-only conversation — that
// keeps the proven qgen/image-gen callers byte-identical to pre-ADR-177. A
// genuinely structured turn (multi-turn, or any function-call /
// function-response / inline-data part) is sent verbatim so the gateway's
// gemini adapter rebuilds the exact conversation + drives tool-calling.
//
// The genai JSON tags (role / parts / text / functionCall / functionResponse /
// inlineData) match the gateway's geminiContent wire shape exactly, so the
// gateway unmarshals contents_json straight into its Vertex request body.
func marshalContents(contents []*genai.Content) (string, error) {
	if !needsStructuredContents(contents) {
		return "", nil
	}
	b, err := json.Marshal(contents)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// needsStructuredContents reports whether the conversation must be carried as
// structured contents_json rather than the flat prompt: true when there is
// more than one content entry OR any part is non-text (functionCall /
// functionResponse / inlineData / fileData) — i.e. anything the flat prompt
// would drop. fileData carries the EPIC-1a batch-grounding object-store  blob reference.
func needsStructuredContents(contents []*genai.Content) bool {
	nonEmpty := 0
	for _, c := range contents {
		if c == nil {
			continue
		}
		nonEmpty++
		for _, p := range c.Parts {
			if p == nil {
				continue
			}
			if p.FunctionCall != nil || p.FunctionResponse != nil || p.InlineData != nil || p.FileData != nil {
				return true
			}
		}
	}
	return nonEmpty > 1
}

// marshalTools serialises the agent's tool declarations (genai []*Tool from
// req.Config.Tools) into the `tools_json` field (ADR-177). The genai-marshaled
// shape (`[{functionDeclarations:[...]}]`) is already Vertex-native, so the
// gateway passes it through verbatim. Returns "" when no tools are declared.
func marshalTools(cfg *genai.GenerateContentConfig) (string, error) {
	if cfg == nil || len(cfg.Tools) == 0 {
		return "", nil
	}
	b, err := json.Marshal(cfg.Tools)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// isTurnInitiating reports whether this Invoke begins a new user-facing turn —
// the request whose LAST content is a plain `user` message (text), NOT a
// tool-result continuation. ADR-177 §5 meters ONLY turn-initiating Invokes so
// one user turn costs one debit, not one-per-tool-loop-iteration. A content
// carrying a FunctionResponse part is the agent feeding a tool result back to
// the model (a continuation) → NOT turn-initiating. An empty conversation is
// treated as turn-initiating (the first call).
func isTurnInitiating(contents []*genai.Content) bool {
	last := lastNonNilContent(contents)
	if last == nil {
		return true
	}
	if last.Role != "" && last.Role != "user" {
		// A trailing model turn is not a user-initiated turn.
		return false
	}
	for _, p := range last.Parts {
		if p != nil && p.FunctionResponse != nil {
			return false // tool-result continuation
		}
	}
	return true
}

func lastNonNilContent(contents []*genai.Content) *genai.Content {
	for i := len(contents) - 1; i >= 0; i-- {
		if contents[i] != nil {
			return contents[i]
		}
	}
	return nil
}

// buildResponseParts assembles the LLMResponse Content parts. When the gateway
// surfaces model-emitted function calls (toolCallsJSON, the genai
// `[{name,args}]` shape), they become genai FunctionCall parts so the ADK
// runtime runs the local tools + continues the loop; any accompanying text is
// appended. With no tool calls it is a single text part (the terminal turn),
// preserving the pre-ADR-177 shape exactly (incl. an empty-completion part).
func buildResponseParts(completion, toolCallsJSON string) ([]*genai.Part, error) {
	if strings.TrimSpace(toolCallsJSON) == "" {
		return []*genai.Part{{Text: completion}}, nil
	}
	var calls []genai.FunctionCall
	if err := json.Unmarshal([]byte(toolCallsJSON), &calls); err != nil {
		return nil, err
	}
	parts := make([]*genai.Part, 0, len(calls)+1)
	for i := range calls {
		fc := calls[i]
		parts = append(parts, &genai.Part{FunctionCall: &fc})
	}
	if completion != "" {
		parts = append(parts, &genai.Part{Text: completion})
	}
	if len(parts) == 0 {
		parts = append(parts, &genai.Part{Text: completion})
	}
	return parts, nil
}

// flattenContents collapses a (potentially multi-turn) ADK Contents slice
// into (prompt, system_prompt). Strategy:
//   - "system" role parts → system_prompt (concatenated with "\n\n").
//   - "user" role parts → prompt (concatenated with "\n\n").
//   - "model" role parts → folded into prompt as `Assistant: ...` lines,
//     since the gateway's Invoke unary surface treats this as a single
//     conversational turn. Multi-turn dialog is preserved via the prompt
//     transcript shape.
//
// This matches the gateway's contract (prompt = REQUIRED user content;
// system_prompt = OPTIONAL platform-controlled instruction; no separate
// multi-turn array).
func flattenContents(contents []*genai.Content) (string, string) {
	var userBuf, sysBuf string
	for _, c := range contents {
		if c == nil {
			continue
		}
		text := joinParts(c.Parts)
		if text == "" {
			continue
		}
		switch c.Role {
		case "system":
			sysBuf = appendBlock(sysBuf, text)
		case "user", "":
			userBuf = appendBlock(userBuf, text)
		case "model", "assistant":
			userBuf = appendBlock(userBuf, "Assistant: "+text)
		default:
			userBuf = appendBlock(userBuf, c.Role+": "+text)
		}
	}
	return userBuf, sysBuf
}

func joinParts(parts []*genai.Part) string {
	var out string
	for _, p := range parts {
		if p == nil || p.Text == "" {
			continue
		}
		out = appendBlock(out, p.Text)
	}
	return out
}

func appendBlock(buf, add string) string {
	if buf == "" {
		return add
	}
	return buf + "\n\n" + add
}

// encodeGenerationConfig extracts the JSON-tagged fields of
// *genai.GenerateContentConfig and packs them into a google.protobuf.Struct
// so the gateway's domain layer can route them to vendor adapters.
//
// We use structpb.NewStruct over a map of the supported fields — the
// gateway tolerates unknown fields (per JSON-via-Struct round-trip) and
// applies the ones it knows.
func encodeGenerationConfig(cfg *genai.GenerateContentConfig) (*structpb.Struct, error) {
	if cfg == nil {
		return nil, nil
	}
	m := map[string]any{}
	if cfg.Temperature != nil {
		m["temperature"] = float64(*cfg.Temperature)
	}
	if cfg.TopP != nil {
		m["top_p"] = float64(*cfg.TopP)
	}
	if cfg.TopK != nil {
		m["top_k"] = float64(*cfg.TopK)
	}
	if cfg.MaxOutputTokens != 0 {
		m["max_output_tokens"] = float64(cfg.MaxOutputTokens)
	}
	if cfg.CandidateCount != 0 {
		m["candidate_count"] = float64(cfg.CandidateCount)
	}
	if cfg.ResponseMIMEType != "" {
		m["response_mime_type"] = cfg.ResponseMIMEType
	}
	if len(cfg.StopSequences) > 0 {
		ss := make([]any, len(cfg.StopSequences))
		for i, s := range cfg.StopSequences {
			ss[i] = s
		}
		m["stop_sequences"] = ss
	}
	if len(m) == 0 {
		return nil, nil
	}
	return structpb.NewStruct(m)
}

// mapFinishDetail translates the gateway's free-form finish_detail string
// into the ADK / genai canonical FinishReason. Unknown values map to
// FinishReasonUnspecified (the iterator-terminal call always sets
// TurnComplete=true regardless).
func mapFinishDetail(s string) genai.FinishReason {
	switch s {
	case "STOP", "stop", "":
		return genai.FinishReasonStop
	case "MAX_TOKENS", "max_tokens":
		return genai.FinishReasonMaxTokens
	case "SAFETY", "safety":
		return genai.FinishReasonSafety
	case "RECITATION", "recitation":
		return genai.FinishReasonRecitation
	default:
		return genai.FinishReasonUnspecified
	}
}

// extractSystemInstruction pulls the instruction text out of
// req.Config.SystemInstruction (genai.Content). The ADK runtime sets
// this from WithInstruction(...) / per-turn InstructionProvider; the
// Vertex Gemini adapter reads it directly off Config. The gateway's
// Invoke RPC carries a flat SystemPrompt string, so we flatten the
// Content's Parts here (text only — non-text parts are ignored since
// the gateway prompt field is string-typed). Empty-safe; returns ""
// for nil Config / nil SystemInstruction / no text parts.
func extractSystemInstruction(cfg *genai.GenerateContentConfig) string {
	if cfg == nil || cfg.SystemInstruction == nil {
		return ""
	}
	var buf string
	for _, p := range cfg.SystemInstruction.Parts {
		if p == nil || p.Text == "" {
			continue
		}
		buf = appendBlock(buf, p.Text)
	}
	return buf
}

// stringSliceToAny widens a []string to []any so it can be embedded in an
// adkmodel.LLMResponse.CustomMetadata map. structpb.NewValue (which ADK
// invokes when stamping the response onto the session event) rejects
// concrete-slice types like []string with "proto: invalid type: []string";
// []any round-trips through structpb.NewList. Always returns a non-nil
// slice (empty []any for nil/empty input) so downstream consumers see a
// stable shape.
func stringSliceToAny(ss []string) []any {
	out := make([]any, len(ss))
	for i, s := range ss {
		out[i] = s
	}
	return out
}

// mergeTokensIntoCompletion injects the gateway's per-hop token counts into
// the completion's top-level JSON object as `input_tokens` / `output_tokens` /
// `tokens_consumed_total` (C3). The Python executor's _map_response reads
// these off the terminal candidate text; the qgen_crew nodes copy them into
// the pipeline_trace rows the W6 token-compounding eval consumes.
//
// Behaviour contract (asserted in client_token_test.go):
//   - nil usage → completion returned UNCHANGED (byte-compatible with pre-C3).
//   - non-JSON-object completion (prose / block message / fenced non-object) →
//     returned UNCHANGED (fail-soft: token injection never corrupts a candidate;
//     the Python side reads tokens_consumed_total=0 on that non-happy path).
//   - JSON-object completion → the three token fields are added at the top
//     level (alongside the existing candidate payload) and re-serialised.
//   - markdown-fenced JSON object (```json … ```) → the fence is peeled, the
//     inner object augmented, and the fence restored so the Python
//     _strip_markdown_fences + _parse_terminal_json path still recovers it.
//
// tokens_consumed_total = input + output (matches the gateway's TokenUsage
// semantics + the orchestrator's existing total-only field).
func mergeTokensIntoCompletion(completion string, usage *mgv1.TokenUsage) string {
	if usage == nil {
		return completion
	}
	body, fence := splitMarkdownFence(completion)
	trimmed := strings.TrimSpace(body)
	// Only merge into a JSON OBJECT — arrays / scalars / prose are left as-is.
	if !strings.HasPrefix(trimmed, "{") {
		return completion
	}
	var obj map[string]any
	if err := json.Unmarshal([]byte(trimmed), &obj); err != nil {
		// Not a clean object (e.g. trailing-extra-char or prompt-echo prefix
		// quirk). Leave verbatim — the Python _parse_terminal_json tolerates
		// those quirks for the candidate itself; we simply forgo the in-band
		// token merge (UsageMetadata + CustomMetadata still carry the counts).
		return completion
	}
	in, out := usage.GetInputTokens(), usage.GetOutputTokens()
	stampTokens(obj, in, out)
	// The generation agent wraps its candidate under a "candidate" key
	// ({"candidate": {...}}); the Python executor's _map_response UNWRAPS that
	// (raw = raw["candidate"]) BEFORE reading tokens_consumed_total. So the
	// token fields must ALSO live inside the nested candidate object to survive
	// the unwrap and be read off the terminal candidate. The critic (and any
	// already-flat candidate) has no wrapper — stamping the top level alone is
	// enough there. Stamp BOTH so either Python shape reads the counts.
	if inner, ok := obj["candidate"].(map[string]any); ok {
		stampTokens(inner, in, out)
	}

	merged, err := json.Marshal(obj)
	if err != nil {
		return completion // unreachable for a map[string]any of JSON scalars
	}
	if fence {
		return "```json\n" + string(merged) + "\n```"
	}
	return string(merged)
}

// stampTokens writes the per-hop token fields onto a decoded JSON object
// (input_tokens / output_tokens / tokens_consumed_total = input + output).
// JSON numbers decode back as float64; we write int64 so a re-marshal emits
// integers (the Python int(...) coercion accepts either, but integers keep the
// wire shape clean).
func stampTokens(obj map[string]any, in, out int64) {
	obj["input_tokens"] = in
	obj["output_tokens"] = out
	obj["tokens_consumed_total"] = in + out
}

// splitMarkdownFence peels a leading ```json / ``` fence (and the trailing
// ```) from s, returning the inner body + whether a fence was present. Mirrors
// the Python _strip_markdown_fences tolerance. When no fence is present the
// input is returned unchanged with fence=false.
func splitMarkdownFence(s string) (body string, fenced bool) {
	t := strings.TrimSpace(s)
	if !strings.HasPrefix(t, "```") {
		return s, false
	}
	// Drop the opening fence line (``` or ```json or ```JSON …).
	nl := strings.IndexByte(t, '\n')
	if nl < 0 {
		return s, false
	}
	inner := t[nl+1:]
	// Drop the trailing closing fence.
	if idx := strings.LastIndex(inner, "```"); idx >= 0 {
		inner = inner[:idx]
	}
	return inner, true
}

// stripMarkdownFence returns the inner body of a fenced string (or the input
// unchanged when no fence is present). Thin wrapper over splitMarkdownFence
// used by tests + any caller that only wants the body.
func stripMarkdownFence(s string) string {
	body, _ := splitMarkdownFence(s)
	return strings.TrimSpace(body)
}
