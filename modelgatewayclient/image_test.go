package modelgatewayclient

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	mgv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/services/model_gateway/v1"
)

type scriptedImageRPC struct {
	answers []func() (*mgv1.InvokeResponse, error)
	reqs    []*mgv1.InvokeRequest
}

func (s *scriptedImageRPC) Invoke(_ context.Context, in *mgv1.InvokeRequest, _ ...grpc.CallOption) (*mgv1.InvokeResponse, error) {
	s.reqs = append(s.reqs, in)
	if len(s.answers) == 0 {
		return nil, status.Error(codes.Internal, "no scripted answer")
	}
	next := s.answers[0]
	s.answers = s.answers[1:]
	return next()
}

func imageCfg() Config {
	return Config{Endpoint: "gw:443", LogicalModelID: "gemini-3-pro-image", FallbackModelIDs: []string{"gemini-2.5-flash-image"},
		AgentID: "qgen_renderer", CrewKind: "qgen", Surface: "qgen", TenantID: "t-default", GCID: "g-default", CallTimeout: time.Second}
}

func newTestInvoker(rpc imageRPC) *ImageInvoker {
	inv := newImageInvokerWith(imageCfg(), rpc, nil, ImageRetryPolicy{Attempts: 3, BaseDelay: time.Millisecond, MaxDelay: 2 * time.Millisecond})
	inv.sleep = func(context.Context, time.Duration) error { return nil }
	return inv
}

func TestImageInvoker_stampsIdentityAndReturnsBytes(t *testing.T) {
	rpc := &scriptedImageRPC{answers: []func() (*mgv1.InvokeResponse, error){
		func() (*mgv1.InvokeResponse, error) {
			return &mgv1.InvokeResponse{ImageBytes: []byte{1, 2, 3}, ImageMimeType: "image/png", ModelVersion: "gemini-3-pro-image-001"}, nil
		},
	}}
	inv := newTestInvoker(rpc)
	res, err := inv.Invoke(context.Background(), ImageRequest{TenantID: "t1", GCID: "g1", Prompt: "a fox on the moon", DispatchIdempotencyKey: "k1"})
	if err != nil || len(res.Bytes) != 3 || res.MimeType != "image/png" || res.ModelVersion != "gemini-3-pro-image-001" || res.Attempts != 1 {
		t.Fatalf("result=%+v err=%v", res, err)
	}
	req := rpc.reqs[0]
	if req.TenantId != "t1" || req.Gcid != "g1" || req.AgentId != "qgen_renderer" || req.Surface != "qgen" || req.CrewKind != "qgen" ||
		req.ResponseModality != ResponseModalityImage || req.LogicalModelId != "gemini-3-pro-image" || len(req.FallbackLogicalModelIds) != 1 ||
		req.DispatchIdempotencyKey != "k1" || req.ActionCode != "" || req.ContentsJson != "" || req.Prompt != "a fox on the moon" {
		t.Fatalf("request not stamped: %+v", req)
	}
	// Defaults apply when the request carries no tenant/gcid.
	rpc.answers = append(rpc.answers, func() (*mgv1.InvokeResponse, error) { return &mgv1.InvokeResponse{ImageBytes: []byte{9}}, nil })
	res, err = inv.Invoke(context.Background(), ImageRequest{Prompt: "p"})
	if err != nil || res.MimeType != "image/png" || rpc.reqs[1].TenantId != "t-default" || rpc.reqs[1].Gcid != "g-default" {
		t.Fatalf("defaults: res=%+v err=%v req=%+v", res, err, rpc.reqs[1])
	}
}

func TestImageInvoker_retriesThrottlingNotContentOrArguments(t *testing.T) {
	throttled := func() (*mgv1.InvokeResponse, error) {
		return nil, status.Error(codes.Unavailable, "invoke vendor_error: gemini: HTTP 429")
	}
	ok := func() (*mgv1.InvokeResponse, error) { return &mgv1.InvokeResponse{ImageBytes: []byte{1}}, nil }
	rpc := &scriptedImageRPC{answers: []func() (*mgv1.InvokeResponse, error){throttled, throttled, ok}}
	res, err := newTestInvoker(rpc).Invoke(context.Background(), ImageRequest{Prompt: "p"})
	if err != nil || res.Attempts != 3 || len(rpc.reqs) != 3 {
		t.Fatalf("throttling must be retried: res=%+v err=%v calls=%d", res, err, len(rpc.reqs))
	}
	// Budget exhausted: the LAST error surfaces.
	rpc = &scriptedImageRPC{answers: []func() (*mgv1.InvokeResponse, error){throttled, throttled, throttled}}
	if _, err := newTestInvoker(rpc).Invoke(context.Background(), ImageRequest{Prompt: "p"}); err == nil || status.Code(err) != codes.Unavailable || len(rpc.reqs) != 3 {
		t.Fatalf("exhausted budget: err=%v calls=%d", err, len(rpc.reqs))
	}
	// A malformed request is not retried.
	rpc = &scriptedImageRPC{answers: []func() (*mgv1.InvokeResponse, error){
		func() (*mgv1.InvokeResponse, error) { return nil, status.Error(codes.InvalidArgument, "bad prompt") }, ok}}
	if _, err := newTestInvoker(rpc).Invoke(context.Background(), ImageRequest{Prompt: "p"}); err == nil || status.Code(err) != codes.InvalidArgument || len(rpc.reqs) != 1 {
		t.Fatalf("invalid argument must fail fast: err=%v calls=%d", err, len(rpc.reqs))
	}
	// No image bytes is a content decision: ErrNoImage, not retried.
	rpc = &scriptedImageRPC{answers: []func() (*mgv1.InvokeResponse, error){
		func() (*mgv1.InvokeResponse, error) { return &mgv1.InvokeResponse{FinishDetail: "SAFETY"}, nil }, ok}}
	if _, err := newTestInvoker(rpc).Invoke(context.Background(), ImageRequest{Prompt: "p"}); !errors.Is(err, ErrNoImage) || !strings.Contains(err.Error(), "SAFETY") || len(rpc.reqs) != 1 {
		t.Fatalf("no image: err=%v calls=%d", err, len(rpc.reqs))
	}
	// An empty prompt is refused before the wire.
	if _, err := newTestInvoker(&scriptedImageRPC{}).Invoke(context.Background(), ImageRequest{}); err == nil {
		t.Fatal("empty prompt must be refused")
	}
}

func TestImageInvoker_editCarriesTheSourceImageInline(t *testing.T) {
	rpc := &scriptedImageRPC{answers: []func() (*mgv1.InvokeResponse, error){
		func() (*mgv1.InvokeResponse, error) {
			return &mgv1.InvokeResponse{ImageBytes: []byte{1}, ImageMimeType: "image/jpeg"}, nil
		}}}
	res, err := newTestInvoker(rpc).Invoke(context.Background(), ImageRequest{Prompt: "make it night", SourceImage: []byte("PNGDATA"), SourceMime: "image/png"})
	if err != nil || res.MimeType != "image/jpeg" {
		t.Fatal(err)
	}
	var contents []map[string]any
	if err := json.Unmarshal([]byte(rpc.reqs[0].ContentsJson), &contents); err != nil || len(contents) != 1 {
		t.Fatalf("contents_json: %v %s", err, rpc.reqs[0].ContentsJson)
	}
	parts := contents[0]["parts"].([]any)
	if len(parts) != 2 || parts[0].(map[string]any)["text"] != "make it night" {
		t.Fatalf("parts: %v", parts)
	}
	inline := parts[1].(map[string]any)["inlineData"].(map[string]any)
	if inline["mimeType"] != "image/png" || inline["data"] != "UE5HREFUQQ==" {
		t.Fatalf("inlineData: %v", inline)
	}
	if rpc.reqs[0].Prompt != "make it night" {
		t.Fatal("the text prompt must still be set beside contents_json")
	}
}

func TestNewImageInvoker_guardsTheConfig(t *testing.T) {
	cfg := imageCfg()
	cfg.Surface = ""
	if _, err := NewImageInvoker(context.Background(), cfg, DefaultImageRetryPolicy()); err == nil || !strings.Contains(err.Error(), "Surface") {
		t.Fatalf("surface guard: %v", err)
	}
	if p := DefaultImageRetryPolicy(); p.Attempts != 4 || p.BaseDelay != 2*time.Second || p.MaxDelay != 30*time.Second {
		t.Fatalf("default policy = %+v", p)
	}
	if !isRetryableImageErr(status.Error(codes.ResourceExhausted, "x")) || isRetryableImageErr(status.Error(codes.PermissionDenied, "x")) || isRetryableImageErr(nil) {
		t.Fatal("retry classification wrong")
	}
}
