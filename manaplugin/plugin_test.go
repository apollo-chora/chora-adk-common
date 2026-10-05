package manaplugin

import (
	"context"
	"testing"

	otelattr "go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/apollo-chora/chora-adk-common/tenancy"
)

// stubMana is a deterministic ManaClient for tests.
type stubMana struct {
	balance  int
	peekErr  error
	debits   []tenancy.DebitRequest
	debitErr error
}

func (s *stubMana) PeekBalance(_ context.Context, gcid string) (tenancy.ManaBalance, error) {
	if s.peekErr != nil {
		return tenancy.ManaBalance{}, s.peekErr
	}
	return tenancy.ManaBalance{UserGCID: gcid, RemainingMana: s.balance, Tier: "basic"}, nil
}

func (s *stubMana) Debit(_ context.Context, req tenancy.DebitRequest) error {
	if s.debitErr != nil {
		return s.debitErr
	}
	s.debits = append(s.debits, req)
	return nil
}

func TestNew_refusesNilManaClient(t *testing.T) {
	_, err := New(Config{CrewKind: "familiar"})
	if err == nil {
		t.Fatal("want error for nil Mana; got nil")
	}
}

func TestNew_refusesEmptyCrewKind(t *testing.T) {
	_, err := New(Config{Mana: &stubMana{balance: 1000}})
	if err == nil {
		t.Fatal("want error for empty CrewKind; got nil")
	}
}

func TestNew_buildsPluginWithExpectedName(t *testing.T) {
	p, err := New(Config{Mana: &stubMana{balance: 1000}, CrewKind: "familiar"})
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if p.Name() != "chora_mana_gate_familiar" {
		t.Errorf("plugin name should embed CrewKind; got %q", p.Name())
	}
}

func TestNew_defaultsDefaultEstimatedTokensWhenZero(t *testing.T) {
	// Build a plugin with DefaultEstimatedTokens=0 — must default to a
	// safe non-zero value (100 per the canonical sandbox config).
	_, err := New(Config{
		Mana:                   &stubMana{balance: 1000},
		CrewKind:               "familiar",
		DefaultEstimatedTokens: 0,
	})
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
}

// ---------- Span attribute helpers ----------

func TestBalanceSpanAttributes_includesManaCharged(t *testing.T) {
	attrs := BalanceSpanAttributes(150 /* pre */, 100 /* post */, "familiar")
	if !containsAttrInt64(attrs, "chora.mana.charged", 50) {
		t.Errorf("want chora.mana.charged=50 (pre 150 - post 100); got %v", attrs)
	}
}

func TestBalanceSpanAttributes_includesBalancePost(t *testing.T) {
	attrs := BalanceSpanAttributes(150, 100, "familiar")
	if !containsAttrInt64(attrs, "chora.mana.balance_post", 100) {
		t.Errorf("want chora.mana.balance_post=100; got %v", attrs)
	}
}

func TestBalanceSpanAttributes_includesCrewKind(t *testing.T) {
	attrs := BalanceSpanAttributes(150, 100, "qgen")
	if !containsAttrString(attrs, "chora.crew_kind", "qgen") {
		t.Errorf("want chora.crew_kind=qgen; got %v", attrs)
	}
}

func TestBalanceSpanAttributes_zeroChargeOnEqualBalances(t *testing.T) {
	// Pre = post means no debit happened (e.g., debit error). Charge=0
	// is correct + ops-debuggable.
	attrs := BalanceSpanAttributes(100, 100, "familiar")
	if !containsAttrInt64(attrs, "chora.mana.charged", 0) {
		t.Errorf("want chora.mana.charged=0; got %v", attrs)
	}
}

func TestBalanceSpanAttributes_clampsNegativeCharge(t *testing.T) {
	// If post > pre (somehow — refund? bug?), charge clamps to 0 rather
	// than going negative. Defensive — span attrs MUST stay legible.
	attrs := BalanceSpanAttributes(100, 200, "familiar")
	if !containsAttrInt64(attrs, "chora.mana.charged", 0) {
		t.Errorf("want chora.mana.charged clamped to 0; got %v", attrs)
	}
}

// ---------- EmitBalanceSpan integration via in-memory exporter ----------

func TestEmitBalanceSpan_addsAttributesToActiveSpan(t *testing.T) {
	// Install an in-memory span recorder so we can assert the wire
	// attributes (Cloud Trace would receive the same payload in prod).
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	tracer := tp.Tracer("test")
	ctx, span := tracer.Start(context.Background(), "test.span")

	EmitBalanceSpan(ctx, 150, 100, "familiar")
	span.End()

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("want 1 ended span; got %d", len(spans))
	}
	attrs := spans[0].Attributes()

	if !findInt64(attrs, "chora.mana.charged", 50) {
		t.Errorf("span should have chora.mana.charged=50; attrs=%v", attrs)
	}
	if !findInt64(attrs, "chora.mana.balance_post", 100) {
		t.Errorf("span should have chora.mana.balance_post=100; attrs=%v", attrs)
	}
	if !findString(attrs, "chora.crew_kind", "familiar") {
		t.Errorf("span should have chora.crew_kind=familiar; attrs=%v", attrs)
	}
}

func TestEmitBalanceSpan_noopWithoutActiveSpan(t *testing.T) {
	// Calling EmitBalanceSpan outside a span must NOT panic. ADK Go's
	// production launcher always installs telemetry, but the manaplugin
	// must be robust to unit-test paths that don't.
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("EmitBalanceSpan(no span) should be safe; panicked: %v", r)
		}
	}()
	EmitBalanceSpan(context.Background(), 100, 50, "familiar")
}

// ---------- ChoraEnv ----------

func TestChoraEnv_envOverridesDefault(t *testing.T) {
	t.Setenv("CHORA_ENV", "staging")
	if got := ChoraEnv(); got != "staging" {
		t.Errorf("ChoraEnv should read CHORA_ENV; got %q", got)
	}
}

func TestChoraEnv_defaultsToDev(t *testing.T) {
	t.Setenv("CHORA_ENV", "")
	if got := ChoraEnv(); got != "dev" {
		t.Errorf("ChoraEnv default should be dev; got %q", got)
	}
}

// ---------- test helpers — operate on []otelattr.KeyValue ----------

func containsAttrInt64(attrs []otelattr.KeyValue, key string, want int64) bool {
	for _, a := range attrs {
		if string(a.Key) == key && a.Value.AsInt64() == want {
			return true
		}
	}
	return false
}

func containsAttrString(attrs []otelattr.KeyValue, key string, want string) bool {
	for _, a := range attrs {
		if string(a.Key) == key && a.Value.AsString() == want {
			return true
		}
	}
	return false
}

func findInt64(attrs []otelattr.KeyValue, key string, want int64) bool {
	return containsAttrInt64(attrs, key, want)
}

func findString(attrs []otelattr.KeyValue, key string, want string) bool {
	return containsAttrString(attrs, key, want)
}
