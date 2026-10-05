package instancedispatch

import (
	"context"
	"errors"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
)

// ---------- New() + NewInstructionProvider() smoke tests ----------

func TestNew_rejectsNilResolver(t *testing.T) {
	_, err := New(Config{StateKey: "familiar_id"})
	if err == nil {
		t.Fatal("want error for nil Resolver; got nil")
	}
}

func TestNew_rejectsEmptyStateKey(t *testing.T) {
	_, err := New(Config{Resolver: staticResolver(InstanceConfig{}, nil)})
	if err == nil {
		t.Fatal("want error for empty StateKey; got nil")
	}
}

func TestNew_returnsPluginWithExpectedName(t *testing.T) {
	p, err := New(Config{
		Resolver: staticResolver(InstanceConfig{}, nil),
		StateKey: "familiar_id",
	})
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if !strings.Contains(p.Name(), "instance_dispatch") {
		t.Errorf("plugin name should describe purpose; got %q", p.Name())
	}
	if !strings.Contains(p.Name(), "familiar_id") {
		t.Errorf("plugin name should embed the state key for ops debuggability; got %q", p.Name())
	}
}

func TestNew_wiresBothBeforeCallbacks(t *testing.T) {
	p, err := New(Config{
		Resolver: staticResolver(InstanceConfig{}, nil),
		StateKey: "familiar_id",
	})
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if p.BeforeRunCallback() == nil {
		t.Error("BeforeRunCallback must be wired (refuses turn when state missing the key)")
	}
	if p.BeforeModelCallback() == nil {
		t.Error("BeforeModelCallback must be wired (filters tools per turn)")
	}
}

func TestNewInstructionProvider_rejectsNilResolver(t *testing.T) {
	_, err := NewInstructionProvider(Config{StateKey: "familiar_id"})
	if err == nil {
		t.Fatal("want error for nil Resolver; got nil")
	}
}

func TestNewInstructionProvider_rejectsEmptyStateKey(t *testing.T) {
	_, err := NewInstructionProvider(Config{Resolver: staticResolver(InstanceConfig{}, nil)})
	if err == nil {
		t.Fatal("want error for empty StateKey; got nil")
	}
}

func TestNewInstructionProvider_returnsCallableProvider(t *testing.T) {
	provider, err := NewInstructionProvider(Config{
		Resolver: staticResolver(InstanceConfig{Instruction: "math prompt"}, nil),
		StateKey: "familiar_id",
	})
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if provider == nil {
		t.Fatal("provider must not be nil on success")
	}

	got, err := provider(newFakeReadonlyCtx("familiar_id", "math-uuid"))
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if got != "math prompt" {
		t.Errorf("provider should return resolver's Instruction; got %q", got)
	}
}

func TestNewInstructionProvider_propagatesResolverError(t *testing.T) {
	wantErr := errors.New("not in registry")
	provider, _ := NewInstructionProvider(Config{
		Resolver: staticResolver(InstanceConfig{}, wantErr),
		StateKey: "familiar_id",
	})

	_, err := provider(newFakeReadonlyCtx("familiar_id", "x"))
	if !errors.Is(err, wantErr) {
		t.Errorf("provider must propagate resolver error; got %v", err)
	}
}

func TestNewInstructionProvider_failsWhenStateMissingKey(t *testing.T) {
	provider, _ := NewInstructionProvider(Config{
		Resolver: staticResolver(InstanceConfig{Instruction: "x"}, nil),
		StateKey: "familiar_id",
	})

	_, err := provider(newFakeReadonlyCtx("", ""))
	if err == nil {
		t.Fatal("want error when state missing the key; got nil")
	}
}

// ---------- BeforeModelCallback integration via plugin ----------

func TestPlugin_BeforeModelCallback_filtersToolsFromState(t *testing.T) {
	// Resolver returns allowed=["atom.search"]; request has 3 tools — only
	// atom.search should survive.
	p, _ := New(Config{
		Resolver: staticResolver(InstanceConfig{AllowedTools: []string{"atom.search"}}, nil),
		StateKey: "familiar_id",
	})

	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{
				{FunctionDeclarations: []*genai.FunctionDeclaration{
					{Name: "atom.search"},
					{Name: "math.solver"},
					{Name: "persona.voice"},
				}},
			},
		},
		Tools: map[string]any{
			"atom.search":   "i1",
			"math.solver":   "i2",
			"persona.voice": "i3",
		},
	}

	cb := p.BeforeModelCallback()
	if cb == nil {
		t.Fatal("BeforeModelCallback should be wired")
	}
	resp, err := cb(newFakeCallbackCtx("familiar_id", "math-uuid"), req)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if resp != nil {
		t.Error("BeforeModelCallback should return nil response (let actual call proceed)")
	}

	if len(req.Tools) != 1 {
		t.Errorf("want 1 tool impl after filter; got %d", len(req.Tools))
	}
	if _, ok := req.Tools["atom.search"]; !ok {
		t.Error("atom.search impl must survive")
	}
	if len(req.Config.Tools[0].FunctionDeclarations) != 1 {
		t.Errorf("want 1 function declaration; got %d", len(req.Config.Tools[0].FunctionDeclarations))
	}
}

func TestPlugin_BeforeModelCallback_returnsErrorWhenStateMissingKey(t *testing.T) {
	p, _ := New(Config{
		Resolver: staticResolver(InstanceConfig{}, nil),
		StateKey: "familiar_id",
	})

	req := &model.LLMRequest{Config: &genai.GenerateContentConfig{}}
	_, err := p.BeforeModelCallback()(newFakeCallbackCtx("", ""), req)
	if err == nil {
		t.Fatal("want error when state missing familiar_id; got nil")
	}
}

func TestPlugin_BeforeRunCallback_returnsErrorWhenStateMissingKey(t *testing.T) {
	// IMDA D3-style discipline: refuse turn rather than silently serve a default.
	p, _ := New(Config{
		Resolver: staticResolver(InstanceConfig{}, nil),
		StateKey: "familiar_id",
	})

	cb := p.BeforeRunCallback()
	if cb == nil {
		t.Fatal("BeforeRunCallback should be wired")
	}
	_, err := cb(newFakeInvocationCtx("", ""))
	if err == nil {
		t.Fatal("want error when state missing familiar_id; got nil")
	}
}

func TestPlugin_BeforeRunCallback_passesWhenResolverSucceeds(t *testing.T) {
	p, _ := New(Config{
		Resolver: staticResolver(InstanceConfig{}, nil),
		StateKey: "familiar_id",
	})

	_, err := p.BeforeRunCallback()(newFakeInvocationCtx("familiar_id", "math-uuid"))
	if err != nil {
		t.Errorf("happy path should not error; got %v", err)
	}
}

func TestPlugin_BeforeRunCallback_returnsErrorWhenResolverFails(t *testing.T) {
	wantErr := errors.New("not in registry")
	p, _ := New(Config{
		Resolver: staticResolver(InstanceConfig{}, wantErr),
		StateKey: "familiar_id",
	})

	_, err := p.BeforeRunCallback()(newFakeInvocationCtx("familiar_id", "ghost-uuid"))
	if !errors.Is(err, wantErr) {
		t.Errorf("BeforeRun should propagate resolver error; got %v", err)
	}
}

// ---------- minimal fakes for agent.* context interfaces ----------

func newFakeReadonlyCtx(key, value string) agent.ReadonlyContext {
	state := map[string]any{}
	if key != "" {
		state[key] = value
	}
	return &fakeReadonlyCtx{ctx: context.Background(), state: state}
}

func newFakeCallbackCtx(key, value string) agent.CallbackContext {
	state := map[string]any{}
	if key != "" {
		state[key] = value
	}
	return &fakeCallbackCtx{fakeReadonlyCtx: fakeReadonlyCtx{ctx: context.Background(), state: state}}
}

func newFakeInvocationCtx(key, value string) agent.InvocationContext {
	state := map[string]any{}
	if key != "" {
		state[key] = value
	}
	return &fakeInvocationCtx{
		ctx:     context.Background(),
		session: &fakeSession{state: &fakeState{m: state}},
	}
}
