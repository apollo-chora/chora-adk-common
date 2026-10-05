package instancedispatch

import (
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
)

// New constructs the dispatch plugin. Wire alongside manaplugin in the
// launcher's PluginConfig.Plugins list.
//
// Plugin name embeds the StateKey for ops debuggability — when multiple
// instance-dispatch plugins are registered (e.g., one for familiar_id and one
// for tenant_id in a future hybrid crew), logs and traces stay legible.
func New(cfg Config) (*plugin.Plugin, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	name := fmt.Sprintf("chora_instance_dispatch_%s", cfg.StateKey)

	return plugin.New(plugin.Config{
		Name: name,
		BeforeRunCallback: func(ic agent.InvocationContext) (*genai.Content, error) {
			_, err := ResolveInstanceFromState(ic, ic.Session().State(), cfg.StateKey, cfg.Resolver)
			if err != nil {
				// Refuse the turn — mirrors manaplugin's tenant_id/user_gcid
				// discipline. The error reaches the caller via an early-exit
				// event (runner.go:218-231) instead of running the agent
				// against a missing or invalid instance config.
				return nil, err
			}
			return nil, nil
		},
		BeforeModelCallback: func(ctx agent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
			instance, err := ResolveInstanceFromState(ctx, ctx.ReadonlyState(), cfg.StateKey, cfg.Resolver)
			if err != nil {
				return nil, err
			}
			ApplyToolFilter(req, instance.AllowedTools)
			return nil, nil
		},
	})
}

// NewInstructionProvider returns an llmagent.InstructionProvider that resolves
// the per-instance instruction from session state on each turn.
//
// Wire the returned provider into `llmagent.Config.InstructionProvider`. Pair
// with New() so the plugin (tool filter) and the provider (instruction) share
// the same Resolver and stay coherent across turns.
//
// The provider replaces the static `Instruction` field — both can be set but
// `InstructionProvider` takes precedence per llmagent.go:209.
func NewInstructionProvider(cfg Config) (llmagent.InstructionProvider, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	return func(ctx agent.ReadonlyContext) (string, error) {
		instance, err := ResolveInstanceFromState(ctx, ctx.ReadonlyState(), cfg.StateKey, cfg.Resolver)
		if err != nil {
			return "", err
		}
		return instance.Instruction, nil
	}, nil
}
