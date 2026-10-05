// Test doubles for terminationplugin. Mirrors the shape used in
// instancedispatch/fakes_test.go — kept self-contained so the plugin
// tests don't drag in the rest of chora-adk-common.

package terminationplugin

import (
	"context"
	"iter"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// fakeReadonlyState implements session.ReadonlyState.
type fakeReadonlyState struct{ m map[string]any }

func (f *fakeReadonlyState) Get(k string) (any, error) {
	if v, ok := f.m[k]; ok {
		return v, nil
	}
	return nil, session.ErrStateKeyNotExist
}

func (f *fakeReadonlyState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range f.m {
			if !yield(k, v) {
				return
			}
		}
	}
}

// fakeState implements session.State.
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

// fakeSession implements session.Session.
type fakeSession struct {
	id    string
	state *fakeState
}

func (f *fakeSession) ID() string                { return f.id }
func (f *fakeSession) AppName() string           { return "fake-app" }
func (f *fakeSession) UserID() string            { return "fake-user" }
func (f *fakeSession) State() session.State      { return f.state }
func (f *fakeSession) Events() session.Events    { return nil }
func (f *fakeSession) LastUpdateTime() time.Time { return time.Time{} }

// fakeInvocationCtx implements agent.InvocationContext.
// Agent() returns nil — the plugin never reads agent.Name() (the AgentID
// is sourced from Config, not the runtime agent).
type fakeInvocationCtx struct {
	ctx     context.Context
	session *fakeSession
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

// fakeCallbackCtx implements agent.CallbackContext.
type fakeCallbackCtx struct {
	ctx   context.Context
	state map[string]any
}

func (f *fakeCallbackCtx) Deadline() (time.Time, bool) { return f.ctx.Deadline() }
func (f *fakeCallbackCtx) Done() <-chan struct{}       { return f.ctx.Done() }
func (f *fakeCallbackCtx) Err() error                  { return f.ctx.Err() }
func (f *fakeCallbackCtx) Value(key any) any           { return f.ctx.Value(key) }
func (f *fakeCallbackCtx) UserContent() *genai.Content { return nil }
func (f *fakeCallbackCtx) InvocationID() string        { return "fake-inv" }
func (f *fakeCallbackCtx) AgentName() string           { return "fake-agent" }
func (f *fakeCallbackCtx) UserID() string              { return "fake-user" }
func (f *fakeCallbackCtx) AppName() string             { return "fake-app" }
func (f *fakeCallbackCtx) SessionID() string           { return "fake-session" }
func (f *fakeCallbackCtx) Branch() string              { return "" }
func (f *fakeCallbackCtx) ReadonlyState() session.ReadonlyState {
	return &fakeReadonlyState{m: f.state}
}
func (f *fakeCallbackCtx) Artifacts() agent.Artifacts { return nil }
func (f *fakeCallbackCtx) State() session.State {
	return &fakeState{m: f.state}
}

// newFakeICtx builds an invocation context preloaded with the given
// session state. agentName is ignored by the plugin (AgentID is sourced
// from Config, not the runtime agent) — accepted for test readability.
func newFakeICtx(_, sessionID string, state map[string]any) *fakeInvocationCtx {
	if state == nil {
		state = map[string]any{}
	}
	return &fakeInvocationCtx{
		ctx:     context.Background(),
		session: &fakeSession{id: sessionID, state: &fakeState{m: state}},
	}
}

func newFakeCallbackCtx(state map[string]any) *fakeCallbackCtx {
	if state == nil {
		state = map[string]any{}
	}
	return &fakeCallbackCtx{ctx: context.Background(), state: state}
}
