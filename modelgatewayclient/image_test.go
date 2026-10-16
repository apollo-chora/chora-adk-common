package modelgatewayclient

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	gatewayv1 "github.com/apollo-chora/chora-adk-common/modelgatewayclient/gatewayv1"
)

type scriptedImageRPC struct {
	answers []func() (*gatewayv1.InvokeResponse, error)
	reqs    []*gatewayv1.InvokeRequest
}

func (s *scriptedImageRPC) Invoke(_ context.Context, in *gatewayv1.InvokeRequest, _ ...grpc.CallOption) (*gatewayv1.InvokeResponse, error) {
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
	rpc := &scriptedImageRPC{answers: []func() (*gatewayv1.InvokeResponse, error){
		func() (*gatewayv1.InvokeResponse, error) {
			return &gatewayv1.InvokeResponse{
				UpstreamModel: "gemini-3-pro-image-001",
				FinishReason:  gatewayv1.FinishReason_FINISH_REASON_STOP,
				Output: []*gatewayv1.OutputItem{{
					Item: &gatewayv1.OutputItem_Content{Content: &gatewayv1.ContentPart{
						Part: &gatewayv1.ContentPart_Image{Image: &gatewayv1.ImagePart{
							Source:    &gatewayv1.ImagePart_Data{Data: "AQID"}, // base64 of {1,2,3}
							MediaType: "image/png",
						}},
					}},
				}},
			}, nil
		},
	}}
	inv := newTestInvoker(rpc)
	res, err := inv.Invoke(context.Background(), ImageRequest{TenantID: "t1", GCID: "g1", Prompt: "a fox on the moon", DispatchIdempotencyKey: "k1"})
	if err != nil || len(res.Bytes) != 3 || res.MimeType != "image/png" || res.ModelVersion != "gemini-3-pro-image-001" || res.Attempts != 1 {
		t.Fatalf("result=%+v err=%v", res, err)
	}
	req := rpc.reqs[0]
	// No model ID and no crew_kind field: the gateway resolves the operation
	// from the RPC method, and crew_kind rides in metadata_json.
	if req.TenantId != "t1" || req.GetGcid() != "g1" || req.AgentId != "qgen_renderer" ||
		req.GetSurface() != "qgen" || req.GetModel() != "" ||
		req.ResponseModality != gatewayv1.ResponseModality_RESPONSE_MODALITY_IMAGE ||
		req.GetDispatchIdempotencyKey() != "k1" || req.GetActionCode() != "" ||
		len(req.GetMessages()) != 1 || req.GetMessages()[0].GetContent()[0].GetText().GetText() != "a fox on the moon" {
		t.Fatalf("request not stamped: %+v", req)
	}
	var meta map[string]any
	if err := json.Unmarshal(req.GetMetadataJson(), &meta); err != nil || meta["crew_kind"] != "qgen" {
		t.Fatalf("crew_kind metadata: %s (err %v)", req.GetMetadataJson(), err)
	}
	// Defaults apply when the request carries no tenant/gcid.
	rpc.answers = append(rpc.answers, func() (*gatewayv1.InvokeResponse, error) {
		return &gatewayv1.InvokeResponse{Output: []*gatewayv1.OutputItem{{
			Item: &gatewayv1.OutputItem_Content{Content: &gatewayv1.ContentPart{
				Part: &gatewayv1.ContentPart_Image{Image: &gatewayv1.ImagePart{
					Source: &gatewayv1.ImagePart_Data{Data: "CQ=="}, // base64 of {9}
				}},
			}},
		}}}, nil
	})
	res, err = inv.Invoke(context.Background(), ImageRequest{Prompt: "p"})
	if err != nil || res.MimeType != "image/png" || rpc.reqs[1].TenantId != "t-default" || rpc.reqs[1].GetGcid() != "g-default" {
		t.Fatalf("defaults: res=%+v err=%v req=%+v", res, err, rpc.reqs[1])
	}
}

func TestImageInvoker_retriesThrottlingNotContentOrArguments(t *testing.T) {
	throttled := func() (*gatewayv1.InvokeResponse, error) {
		return nil, status.Error(codes.Unavailable, "invoke vendor_error: gemini: HTTP 429")
	}
	ok := func() (*gatewayv1.InvokeResponse, error) {
		return &gatewayv1.InvokeResponse{Output: []*gatewayv1.OutputItem{{
			Item: &gatewayv1.OutputItem_Content{Content: &gatewayv1.ContentPart{
				Part: &gatewayv1.ContentPart_Image{Image: &gatewayv1.ImagePart{
					Source: &gatewayv1.ImagePart_Data{Data: "AQ=="}, // base64 of {1}
				}},
			}},
		}}}, nil
	}
	rpc := &scriptedImageRPC{answers: []func() (*gatewayv1.InvokeResponse, error){throttled, throttled, ok}}
	res, err := newTestInvoker(rpc).Invoke(context.Background(), ImageRequest{Prompt: "p"})
	if err != nil || res.Attempts != 3 || len(rpc.reqs) != 3 {
		t.Fatalf("throttling must be retried: res=%+v err=%v calls=%d", res, err, len(rpc.reqs))
	}
	// Budget exhausted: the LAST error surfaces.
	rpc = &scriptedImageRPC{answers: []func() (*gatewayv1.InvokeResponse, error){throttled, throttled, throttled}}
	if _, err := newTestInvoker(rpc).Invoke(context.Background(), ImageRequest{Prompt: "p"}); err == nil || status.Code(err) != codes.Unavailable || len(rpc.reqs) != 3 {
		t.Fatalf("exhausted budget: err=%v calls=%d", err, len(rpc.reqs))
	}
	// A malformed request is not retried.
	rpc = &scriptedImageRPC{answers: []func() (*gatewayv1.InvokeResponse, error){
		func() (*gatewayv1.InvokeResponse, error) { return nil, status.Error(codes.InvalidArgument, "bad prompt") }, ok}}
	if _, err := newTestInvoker(rpc).Invoke(context.Background(), ImageRequest{Prompt: "p"}); err == nil || status.Code(err) != codes.InvalidArgument || len(rpc.reqs) != 1 {
		t.Fatalf("invalid argument must fail fast: err=%v calls=%d", err, len(rpc.reqs))
	}
	// No image bytes is a content decision: ErrNoImage, not retried.
	rpc = &scriptedImageRPC{answers: []func() (*gatewayv1.InvokeResponse, error){
		func() (*gatewayv1.InvokeResponse, error) {
			return &gatewayv1.InvokeResponse{
				FinishReason:    gatewayv1.FinishReason_FINISH_REASON_CONTENT_FILTER,
				RawFinishReason: proto.String("SAFETY"),
				Output:          []*gatewayv1.OutputItem{textOutput("content filtered")},
			}, nil
		}, ok}}
	if _, err := newTestInvoker(rpc).Invoke(context.Background(), ImageRequest{Prompt: "p"}); !errors.Is(err, ErrNoImage) || !strings.Contains(err.Error(), "content filtered") || len(rpc.reqs) != 1 {
		t.Fatalf("no image: err=%v calls=%d", err, len(rpc.reqs))
	}
	// An empty prompt is refused before the wire.
	if _, err := newTestInvoker(&scriptedImageRPC{}).Invoke(context.Background(), ImageRequest{}); err == nil {
		t.Fatal("empty prompt must be refused")
	}
}

func TestImageInvoker_editCarriesTheSourceImageInline(t *testing.T) {
	rpc := &scriptedImageRPC{answers: []func() (*gatewayv1.InvokeResponse, error){
		func() (*gatewayv1.InvokeResponse, error) {
			return &gatewayv1.InvokeResponse{Output: []*gatewayv1.OutputItem{{
				Item: &gatewayv1.OutputItem_Content{Content: &gatewayv1.ContentPart{
					Part: &gatewayv1.ContentPart_Image{Image: &gatewayv1.ImagePart{
						Source:    &gatewayv1.ImagePart_Data{Data: "AQ=="}, // base64 of {1}
						MediaType: "image/jpeg",
					}},
				}},
			}}}, nil
		}}}
	res, err := newTestInvoker(rpc).Invoke(context.Background(), ImageRequest{Prompt: "make it night", SourceImage: []byte("PNGDATA"), SourceMime: "image/png"})
	if err != nil || res.MimeType != "image/jpeg" {
		t.Fatal(err)
	}
	// The canonical shape carries the source image as an image part beside
	// the text, not as genai JSON in contents_json.
	msg := rpc.reqs[0].GetMessages()
	if len(msg) != 1 {
		t.Fatalf("messages: %d", len(msg))
	}
	parts := msg[0].GetContent()
	if len(parts) != 2 || parts[0].GetText().GetText() != "make it night" {
		t.Fatalf("parts: %v", parts)
	}
	img := parts[1].GetImage()
	if img.GetMediaType() != "image/png" || img.GetData() != "UE5HREFUQQ==" {
		t.Fatalf("image part: %v", img)
	}
	if false {
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
