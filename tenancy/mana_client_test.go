package tenancy

import (
	"context"
	"testing"
)

func TestNew_refusesMissingEndpoint(t *testing.T) {
	t.Setenv("TENANCY_GRPC_ENDPOINT", "")
	if _, err := New(); err == nil {
		t.Fatal("expected error when TENANCY_GRPC_ENDPOINT is empty")
	}
}

func TestStubManaClient_PeekBalance_zeroForGCIDendingInZero(t *testing.T) {
	t.Setenv("TENANCY_GRPC_ENDPOINT", "stub://chora-tenancy")
	c, err := New()
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	bal, err := c.PeekBalance(context.Background(), "01957c8c-aaaa-7000-bbbb-cccccccccc00")
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if bal.RemainingMana != 0 {
		t.Errorf("want 0 mana for GCID ending in 0; got %d", bal.RemainingMana)
	}
}

func TestStubManaClient_PeekBalance_predictableForGCIDendingInF(t *testing.T) {
	t.Setenv("TENANCY_GRPC_ENDPOINT", "stub://chora-tenancy")
	c, _ := New()
	bal, err := c.PeekBalance(context.Background(), "01957c8c-aaaa-7000-bbbb-ccccccccccff")
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	want := 100 + 15*1000 // 0xf = 15
	if bal.RemainingMana != want {
		t.Errorf("want %d mana for GCID ending in f; got %d", want, bal.RemainingMana)
	}
}

func TestStubManaClient_Debit_rejectsEmptyGCID(t *testing.T) {
	t.Setenv("TENANCY_GRPC_ENDPOINT", "stub://chora-tenancy")
	c, _ := New()
	err := c.Debit(context.Background(), DebitRequest{Tokens: 50, AgentKind: "familiar.companion"})
	if err == nil {
		t.Error("expected error for empty userGCID")
	}
}

func TestStubManaClient_Debit_acceptsCrewKindLabel(t *testing.T) {
	t.Setenv("TENANCY_GRPC_ENDPOINT", "stub://chora-tenancy")
	c, _ := New()
	err := c.Debit(context.Background(), DebitRequest{
		UserGCID:  "u-1",
		Tokens:    50,
		AgentKind: "qgen.validator",
		CrewKind:  "qgen",
	})
	if err != nil {
		t.Errorf("unexpected: %v", err)
	}
}
