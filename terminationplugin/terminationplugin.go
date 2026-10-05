// Package terminationplugin wires structured AgentTerminated emission
// into the ADK Go agent run lifecycle.
//
// W3 foundation Phase 4 (2026-05-12). Every Chora ADK Go agent
// (Familiar, Multi-Familiar, QGen, future runtimes) registers this
// plugin alongside manaplugin + instancedispatch so the entry-boundary
// produces exactly one “chora.ai_kernel.agent.terminated.v1“ event per
// execution lifecycle, regardless of success/failure outcome.
//
// Contract:
//
//	BeforeRun                                       → clear stale error capture
//	(model call OK + no tool error + no run error)  → AfterRun emits SUCCESS
//	OnModelError fires                              → captured;  AfterRun emits RUNTIME_ERROR
//	OnToolError fires                               → captured;  AfterRun emits RUNTIME_ERROR
//
// Errors from Publisher.Publish are LOGGED and SWALLOWED — the agent
// run termination is past the rescue point; we never panic. The
// publisher is expected to be durable (event bus or outbox-backed
// per agentic-resilience-d6 skill Pillar 2) so transient publish
// failures land on a retry path rather than vanishing.
//
// State keys (namespaced under chora.termination.*) carry the
// per-run capture across callbacks. They are reset on BeforeRun.
package terminationplugin

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// AgentTerminationCode enum mirrors chora.ai_kernel.v1.AgentTerminationCode.
const (
	CodeSuccess           = "AGENT_TERMINATION_CODE_SUCCESS"
	CodeMaxIterations     = "AGENT_TERMINATION_CODE_MAX_ITERATIONS"
	CodeToolFailed        = "AGENT_TERMINATION_CODE_TOOL_FAILED"
	CodeLLMError          = "AGENT_TERMINATION_CODE_LLM_ERROR"
	CodeTimeout           = "AGENT_TERMINATION_CODE_TIMEOUT"
	CodeGuardrailRejected = "AGENT_TERMINATION_CODE_GUARDRAIL_REJECTED"
	CodeRuntimeError      = "AGENT_TERMINATION_CODE_RUNTIME_ERROR"
	CodeRuntimeTerminated = "AGENT_TERMINATION_CODE_RUNTIME_TERMINATED"
	CodeUnknown           = "AGENT_TERMINATION_CODE_UNKNOWN"
)

// Session-state keys for per-run capture. Namespaced under
// chora.termination.* so they don't collide with other plugins'
// state. Exported as untyped strings for test access.
const (
	stateKeyLastError      = "chora.termination.last_error"
	stateKeyLastErrorStep  = "chora.termination.last_error_step"
	stateKeyLastErrorTool  = "chora.termination.last_error_tool"
	stateKeyIterationCount = "chora.termination.iteration_count"
)

// AgentTerminatedEvent carries the fields of
// chora.ai_kernel.v1.AgentTerminated. Producers (this plugin) build it
// and hand it to the Publisher; publishers translate to outbox row /
// event message / etc.
type AgentTerminatedEvent struct {
	AgentID          string
	AgentAGID        string
	ExecutionID      string
	Runtime          string
	TerminationCode  string
	CrewID           string
	CrewPattern      string
	TenantID         string
	GCID             string
	LastStateNode    string
	LastToolName     string
	LastErrorMessage string
	IterationCount   int
	PartialState     map[string]string
	CurrentSpanID    string
	TerminatedAt     time.Time
	Traceparent      string
	Tracestate       string
}

// Publisher is the port the plugin uses to emit AgentTerminated. The
// adapter is service-specific (event bus direct vs HTTP-to-
// orchestrator outbox vs local outbox). The plugin doesn't care.
type Publisher interface {
	Publish(ctx context.Context, event AgentTerminatedEvent) error
}

// Config configures the plugin.
type Config struct {
	// Publisher emits AgentTerminated. REQUIRED.
	Publisher Publisher

	// AgentID is the agent role/type name (NOT the per-instance AGID).
	// Examples: "familiar_companion", "qgen_p2_generator",
	// "qgen_p2_evaluator". REQUIRED.
	AgentID string

	// AgentAGID is the per-instance AGID (per ADR-132). Optional —
	// system-owned agents that aren't published as A2A agents leave
	// this empty.
	AgentAGID string

	// Runtime tells subscribers which runtime the terminating agent
	// uses. One of AGENT_EXECUTION_RUNTIME_*. REQUIRED.
	Runtime string

	// CrewKind / CrewPattern stamp the crew context on the event.
	// CrewKind is mapped into CrewID on the emitted event for chaos-
	// test forensics (per crew-composition skill, CrewID is the
	// joinable correlation column for cross-agent traces).
	CrewKind    string
	CrewPattern string

	// MaxIterations is the hard cap on model calls per agent run
	// (W3 foundation Phase 6, Option A — counter-via-plugin). The
	// plugin increments a per-run counter in session state on every
	// BeforeModelCallback invocation; when the counter EXCEEDS
	// MaxIterations the callback returns an error that aborts the
	// run, and AfterRun emits AgentTerminated with code
	// MAX_ITERATIONS.
	//
	// 0 = unlimited. Per user direction 2026-05-12 the operational
	// default for POC chaos testing is 3; Familiar + QGen main.go
	// set this explicitly. Production may raise via agent_card.yaml.
	//
	// Why this lives in terminationplugin rather than its own plugin:
	// the cap and the structured terminate-event share state machine
	// (the last_error_step marker drives MAX_ITERATIONS vs RUNTIME_ERROR
	// disambiguation in AfterRun). Splitting them would require a
	// cross-plugin protocol.
	MaxIterations int

	// Logger overrides the default slog logger.
	Logger *slog.Logger
}

func (c Config) validate() error {
	if c.Publisher == nil {
		return errors.New("terminationplugin: Publisher required")
	}
	if c.AgentID == "" {
		return errors.New("terminationplugin: AgentID required")
	}
	if c.Runtime == "" {
		return errors.New("terminationplugin: Runtime required")
	}
	return nil
}

// New constructs the termination plugin. Returns an error if Config is
// incomplete. The plugin name embeds AgentID for ops debuggability
// when multiple agents register independently.
func New(cfg Config) (*plugin.Plugin, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}
	name := fmt.Sprintf("chora_termination_%s", cfg.AgentID)

	return plugin.New(plugin.Config{
		Name: name,

		BeforeRunCallback: func(ic agent.InvocationContext) (*genai.Content, error) {
			// Clear stale per-run capture — a saga that ran twice on
			// the same session must not inherit the previous run's
			// error state.
			s := ic.Session().State()
			if err := s.Set(stateKeyLastError, ""); err != nil {
				logger.Debug("clear last_error failed", "err", err)
			}
			if err := s.Set(stateKeyLastErrorStep, ""); err != nil {
				logger.Debug("clear last_error_step failed", "err", err)
			}
			if err := s.Set(stateKeyLastErrorTool, ""); err != nil {
				logger.Debug("clear last_error_tool failed", "err", err)
			}
			if err := s.Set(stateKeyIterationCount, 0); err != nil {
				logger.Debug("clear iteration_count failed", "err", err)
			}
			return nil, nil
		},

		BeforeModelCallback: func(cbCtx agent.CallbackContext, _ *model.LLMRequest) (*model.LLMResponse, error) {
			// Increment per-run iteration counter regardless of cap —
			// it lands on the emitted AgentTerminated event for
			// forensics.
			s := cbCtx.State()
			n := getInt(s, stateKeyIterationCount) + 1
			_ = s.Set(stateKeyIterationCount, n)
			if cfg.MaxIterations > 0 && n > cfg.MaxIterations {
				// Mark the failure mode so AfterRun emits the
				// MAX_ITERATIONS code rather than the generic
				// RUNTIME_ERROR.
				_ = s.Set(stateKeyLastError,
					fmt.Sprintf("max_iterations exceeded: cap=%d, count=%d",
						cfg.MaxIterations, n))
				_ = s.Set(stateKeyLastErrorStep, "max_iterations")
				return nil, fmt.Errorf(
					"terminationplugin: max_iterations exceeded (cap=%d, count=%d)",
					cfg.MaxIterations, n)
			}
			return nil, nil
		},

		OnModelErrorCallback: func(cbCtx agent.CallbackContext, _ *model.LLMRequest, llmErr error) (*model.LLMResponse, error) {
			captureErrorOnState(cbCtx.State(), "model_call", "", llmErr)
			// Do NOT swallow the upstream error; let the runner decide
			// retry semantics. The plugin only observes.
			return nil, nil
		},

		OnToolErrorCallback: func(cbCtx tool.Context, t tool.Tool, _ map[string]any, toolErr error) (map[string]any, error) {
			name := ""
			if t != nil {
				name = t.Name()
			}
			captureErrorOnState(cbCtx.State(), "tool_call", name, toolErr)
			return nil, nil
		},

		AfterRunCallback: func(ic agent.InvocationContext) {
			emit(ic, cfg, logger)
		},
	})
}

// emit builds the AgentTerminatedEvent + hands it to the publisher.
// Errors from the publisher are logged + swallowed.
func emit(ic agent.InvocationContext, cfg Config, logger *slog.Logger) {
	s := ic.Session().State()
	tenantID := getString(s, "tenant_id")
	gcid := getString(s, "user_gcid")
	lastError := getString(s, stateKeyLastError)
	lastErrorStep := getString(s, stateKeyLastErrorStep)
	lastErrorTool := getString(s, stateKeyLastErrorTool)
	iter := getInt(s, stateKeyIterationCount)

	code := CodeSuccess
	switch {
	case lastErrorStep == "max_iterations":
		code = CodeMaxIterations
	case lastError != "":
		code = CodeRuntimeError
	}

	event := AgentTerminatedEvent{
		AgentID:          cfg.AgentID,
		AgentAGID:        cfg.AgentAGID,
		ExecutionID:      executionIDOf(ic),
		Runtime:          cfg.Runtime,
		TerminationCode:  code,
		CrewID:           cfg.CrewKind,
		CrewPattern:      cfg.CrewPattern,
		TenantID:         tenantID,
		GCID:             gcid,
		LastStateNode:    lastErrorStep,
		LastToolName:     lastErrorTool,
		LastErrorMessage: trunc(lastError, 300),
		IterationCount:   iter,
		TerminatedAt:     time.Now().UTC(),
	}

	if err := cfg.Publisher.Publish(ic, event); err != nil {
		logger.Warn(
			"terminationplugin_emit_failed",
			slog.String("agent_id", cfg.AgentID),
			slog.String("execution_id", event.ExecutionID),
			slog.String("termination_code", event.TerminationCode),
			slog.String("error", trunc(err.Error(), 300)),
		)
	}
}

// StateKeyDispatchExecutionID is the session-state key agentdispatch stamps
// with the dispatch's execution id (agentdispatch.StateKeyDispatchExecutionID;
// duplicated here so this package stays import-free of agentdispatch).
const StateKeyDispatchExecutionID = "dispatch_execution_id"

// executionIDOf names the run on the emitted event: the dispatch's execution
// id when the session state carries one (ADR-254: on a persistent
// conversation the session id is the CONVERSATION, which outlives the
// dispatch and would mis-correlate every turn), else the session id, which
// on every per-dispatch session is "dispatch:<execution id>" already.
func executionIDOf(ic agent.InvocationContext) string {
	if id := getString(ic.Session().State(), StateKeyDispatchExecutionID); id != "" {
		return id
	}
	return ic.Session().ID()
}

// captureErrorOnState records the last error + the step it came from.
// Exported variant CaptureToolErrorOnState exposes the same surface
// to tests + adapters that wrap tool errors outside the SDK callback
// (rare in production).
func captureErrorOnState(s session.State, step, toolName string, err error) {
	if err == nil {
		return
	}
	_ = s.Set(stateKeyLastError, err.Error())
	_ = s.Set(stateKeyLastErrorStep, step)
	if toolName != "" {
		_ = s.Set(stateKeyLastErrorTool, toolName)
	}
}

// CaptureToolErrorOnState is the public test/adapter helper for the
// tool-error capture path. Production code wires this via the
// OnToolErrorCallback above; this entrypoint allows direct injection
// without constructing a tool.Context.
func CaptureToolErrorOnState(s session.State, toolName string, err error) {
	captureErrorOnState(s, "tool_call", toolName, err)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func getString(s interface {
	Get(string) (any, error)
}, key string) string {
	v, err := s.Get(key)
	if err != nil {
		return ""
	}
	if str, ok := v.(string); ok {
		return str
	}
	return ""
}

func getInt(s interface {
	Get(string) (any, error)
}, key string) int {
	v, err := s.Get(key)
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
	}
	return 0
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

// LoggingPublisher is a POC-scope Publisher that writes the
// AgentTerminated event as a structured slog record. Sandbox /
// pre-event-bus-wiring use only — production wires a real event bus
// (or outbox-backed) Publisher. Operators can grep
// `agent_terminated_emit` in the service log to observe terminations
// during the multi-crew chaos test even before the wire-level publisher
// is plumbed.
type LoggingPublisher struct {
	Logger *slog.Logger
}

// Publish emits an INFO log line carrying all AgentTerminated fields
// as structured attributes. Never returns an error (logging cannot
// fail in a way the plugin can react to).
func (l *LoggingPublisher) Publish(_ context.Context, e AgentTerminatedEvent) error {
	logger := l.Logger
	if logger == nil {
		logger = slog.Default()
	}
	logger.Info(
		"agent_terminated_emit",
		slog.String("agent_id", e.AgentID),
		slog.String("agent_agid", e.AgentAGID),
		slog.String("execution_id", e.ExecutionID),
		slog.String("runtime", e.Runtime),
		slog.String("termination_code", e.TerminationCode),
		slog.String("crew_id", e.CrewID),
		slog.String("crew_pattern", e.CrewPattern),
		slog.String("tenant_id", e.TenantID),
		slog.String("gcid", e.GCID),
		slog.String("last_state_node", e.LastStateNode),
		slog.String("last_tool_name", e.LastToolName),
		slog.String("last_error_message", e.LastErrorMessage),
		slog.Int("iteration_count", e.IterationCount),
		slog.String("current_span_id", e.CurrentSpanID),
		slog.Time("terminated_at", e.TerminatedAt),
	)
	return nil
}

// Compile-time: ensure llmagent + tool symbols stay imported (they're
// only referenced inside the New() closure literals — Go's import-
// usage tracker handles that, but the explicit underscore here makes
// the dependency relationship obvious to readers.)
var (
	_           = llmagent.BeforeModelCallback(nil)
	_           = (tool.Context)(nil)
	_ Publisher = (*LoggingPublisher)(nil)
)
