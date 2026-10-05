package instancedispatch

// Shared fakes for plugin_test.go callback tests. Kept in a separate file so
// the pure-function tests (instancedispatch_test.go) stay free of context
// plumbing.

import (
	"context"
	"iter"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// ---------- fakeReadonlyState (implements session.ReadonlyState) ----------

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

// ---------- fakeState (implements session.State — read+write) ----------

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

// ---------- fakeSession ----------

type fakeSession struct{ state *fakeState }

func (f *fakeSession) ID() string                { return "fake-session" }
func (f *fakeSession) AppName() string           { return "fake-app" }
func (f *fakeSession) UserID() string            { return "fake-user" }
func (f *fakeSession) State() session.State      { return f.state }
func (f *fakeSession) Events() session.Events    { return nil }
func (f *fakeSession) LastUpdateTime() time.Time { return time.Time{} }

// ---------- fakeReadonlyCtx (implements agent.ReadonlyContext) ----------

type fakeReadonlyCtx struct {
	ctx   context.Context
	state map[string]any
}

func (f *fakeReadonlyCtx) Deadline() (time.Time, bool) { return f.ctx.Deadline() }
func (f *fakeReadonlyCtx) Done() <-chan struct{}       { return f.ctx.Done() }
func (f *fakeReadonlyCtx) Err() error                  { return f.ctx.Err() }
func (f *fakeReadonlyCtx) Value(key any) any           { return f.ctx.Value(key) }
func (f *fakeReadonlyCtx) UserContent() *genai.Content { return nil }
func (f *fakeReadonlyCtx) InvocationID() string        { return "fake-inv" }
func (f *fakeReadonlyCtx) AgentName() string           { return "fake-agent" }
func (f *fakeReadonlyCtx) UserID() string              { return "fake-user" }
func (f *fakeReadonlyCtx) AppName() string             { return "fake-app" }
func (f *fakeReadonlyCtx) SessionID() string           { return "fake-session" }
func (f *fakeReadonlyCtx) Branch() string              { return "" }
func (f *fakeReadonlyCtx) ReadonlyState() session.ReadonlyState {
	return &fakeReadonlyState{m: f.state}
}

// ---------- fakeCallbackCtx (implements agent.CallbackContext) ----------

type fakeCallbackCtx struct {
	fakeReadonlyCtx
}

func (f *fakeCallbackCtx) Artifacts() agent.Artifacts { return nil }
func (f *fakeCallbackCtx) State() session.State {
	return &fakeState{m: f.fakeReadonlyCtx.state}
}

// ---------- fakeInvocationCtx (implements agent.InvocationContext) ----------

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
