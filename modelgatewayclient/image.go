package modelgatewayclient

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"strings"
	"time"

	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	mgv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/services/model_gateway/v1"
)

// ResponseModalityImage is the InvokeRequest.response_modality value that
// asks the gateway for an image (InvokeResponse.image_bytes / image_mime_type).
const ResponseModalityImage = "IMAGE"

// ErrNoImage is returned when the gateway answered without image bytes: a
// CONTENT decision (Model Armor or vendor safety block, a text-only answer),
// never a transport fault, so it is not retried.
var ErrNoImage = errors.New("modelgatewayclient: gateway returned no image bytes")

// ImageRetryPolicy bounds the retry on a THROTTLED image model.
// gemini-3-pro-image runs on Vertex dynamic shared quota: a 429 is capacity
// throttling no quota grant can raise (the gateway surfaces it as UNAVAILABLE
// wrapping the vendor HTTP 429), and the documented client contract is
// exponential backoff with jitter. Non-retryable codes fail on the first
// attempt so a malformed or unauthorised request never burns the budget.
type ImageRetryPolicy struct {
	Attempts  int
	BaseDelay time.Duration
	MaxDelay  time.Duration
}

// DefaultImageRetryPolicy mirrors the kennel's image client (4 attempts,
// 2s base, 30s cap, full jitter).
func DefaultImageRetryPolicy() ImageRetryPolicy {
	return ImageRetryPolicy{Attempts: 4, BaseDelay: 2 * time.Second, MaxDelay: 30 * time.Second}
}

// imageRPC is the slice of the gateway stub the invoker uses (a test injects
// a fake; production passes the generated client).
type imageRPC interface {
	Invoke(ctx context.Context, in *mgv1.InvokeRequest, opts ...grpc.CallOption) (*mgv1.InvokeResponse, error)
}

// ImageInvoker calls ModelGatewayService/Invoke with response_modality=IMAGE
// (the same call the kennel's render node made before ADR-254 moved the scene
// render onto the qgen_render lane). It shares the gateway dial, identity and
// D7 stamping with the ADK LLM wrapper but is not an adkmodel.LLM: a render is
// not a chat turn.
type ImageInvoker struct {
	cfg   Config
	conn  *grpc.ClientConn
	grpc  imageRPC
	retry ImageRetryPolicy
	sleep func(context.Context, time.Duration) error
}

// NewImageInvoker dials the gateway for image-modality Invokes. The same
// Config guards apply (Endpoint, LogicalModelID = the image model, AgentID,
// Surface, TenantID, GCID); CallTimeout defaults to 120s, the per-attempt
// ceiling measured for gemini-3-pro-image.
func NewImageInvoker(ctx context.Context, cfg Config, retry ImageRetryPolicy) (*ImageInvoker, error) {
	if err := validateConfig(&cfg); err != nil {
		return nil, err
	}
	if cfg.CallTimeout == 0 || cfg.CallTimeout == 90*time.Second {
		cfg.CallTimeout = 120 * time.Second
	}
	conn, err := dialGateway(ctx, cfg)
	if err != nil {
		return nil, err
	}
	return newImageInvokerWith(cfg, mgv1.NewModelGatewayServiceClient(conn), conn, retry), nil
}

func newImageInvokerWith(cfg Config, rpc imageRPC, conn *grpc.ClientConn, retry ImageRetryPolicy) *ImageInvoker {
	if retry.Attempts < 1 {
		retry.Attempts = 1
	}
	if retry.MaxDelay < retry.BaseDelay {
		retry.MaxDelay = retry.BaseDelay
	}
	return &ImageInvoker{cfg: cfg, conn: conn, grpc: rpc, retry: retry, sleep: sleepCtx}
}

func validateConfig(cfg *Config) error {
	switch {
	case cfg.Endpoint == "":
		return errors.New("modelgatewayclient: Endpoint required")
	case cfg.LogicalModelID == "":
		return errors.New("modelgatewayclient: LogicalModelID required")
	case cfg.AgentID == "":
		return errors.New("modelgatewayclient: AgentID required")
	case cfg.Surface == "":
		return errors.New("modelgatewayclient: Surface required (the crew id, ADR-254 D7)")
	case cfg.TenantID == "":
		return errors.New("modelgatewayclient: TenantID required (CHORA_GATEWAY_TENANT_ID)")
	case cfg.GCID == "":
		return errors.New("modelgatewayclient: GCID required (CHORA_GATEWAY_GCID)")
	}
	if cfg.Audience == "" {
		cfg.Audience = "https://gateway.chora.site"
	}
	return nil
}

// Close releases the connection.
func (i *ImageInvoker) Close() error {
	if i.conn == nil {
		return nil
	}
	return i.conn.Close()
}

// ImageRequest is one image-modality Invoke.
type ImageRequest struct {
	// TenantID / GCID are the REQUESTING tenant and learner (the dispatch
	// envelope's); empty falls back to the client's configured defaults.
	TenantID string
	GCID     string
	// Prompt is the scene description (REQUIRED; the gateway rejects an empty
	// prompt before it inspects contents_json).
	Prompt string
	// SourceImage, when set, makes this an EDIT (ADR-210 image-to-image): the
	// original rides as an inlineData part beside the prompt.
	SourceImage []byte
	SourceMime  string
	// DispatchIdempotencyKey (R22) makes a redelivered dispatch a keyed claim.
	DispatchIdempotencyKey string
	// ActionCode is the metering code; "" for an un-metered call (D7).
	ActionCode string
}

// ImageResult is the gateway's image answer.
type ImageResult struct {
	Bytes        []byte
	MimeType     string
	ModelVersion string
	Attempts     int
}

// Invoke performs the image Invoke with the bounded retry.
func (i *ImageInvoker) Invoke(ctx context.Context, req ImageRequest) (ImageResult, error) {
	if strings.TrimSpace(req.Prompt) == "" {
		return ImageResult{}, errors.New("modelgatewayclient: image prompt is required")
	}
	tenantID, gcid := i.cfg.TenantID, i.cfg.GCID
	if t := strings.TrimSpace(req.TenantID); t != "" {
		tenantID = t
	}
	if g := strings.TrimSpace(req.GCID); g != "" {
		gcid = g
	}
	contentsJSON := ""
	if len(req.SourceImage) > 0 {
		contentsJSON = inlineImageContentsJSON(req.Prompt, req.SourceImage, req.SourceMime)
	}
	invokeReq := &mgv1.InvokeRequest{
		TenantId:                tenantID,
		Gcid:                    gcid,
		AgentId:                 i.cfg.AgentID,
		CrewKind:                i.cfg.CrewKind,
		LogicalModelId:          i.cfg.LogicalModelID,
		FallbackLogicalModelIds: i.cfg.FallbackModelIDs,
		Prompt:                  req.Prompt,
		ResponseModality:        ResponseModalityImage,
		ContentsJson:            contentsJSON,
		ActionCode:              req.ActionCode,
		Surface:                 i.cfg.Surface,
		DispatchIdempotencyKey:  req.DispatchIdempotencyKey,
	}
	if sc := trace.SpanContextFromContext(ctx); sc.IsValid() {
		invokeReq.Traceparent = fmt.Sprintf("00-%s-%s-%02x", sc.TraceID(), sc.SpanID(), byte(sc.TraceFlags()))
		if tsv := sc.TraceState().String(); tsv != "" {
			invokeReq.Tracestate = tsv
		}
	}
	var lastErr error
	for attempt := 1; attempt <= i.retry.Attempts; attempt++ {
		callCtx, cancel := context.WithTimeout(ctx, i.cfg.CallTimeout)
		resp, err := i.grpc.Invoke(callCtx, invokeReq)
		cancel()
		if err == nil {
			if len(resp.GetImageBytes()) == 0 {
				detail := resp.GetFinishDetail()
				if detail == "" {
					detail = resp.GetCompletion()
				}
				if len(detail) > 200 {
					detail = detail[:200]
				}
				return ImageResult{Attempts: attempt}, fmt.Errorf("%w (detail=%q)", ErrNoImage, detail)
			}
			mime := resp.GetImageMimeType()
			if mime == "" {
				mime = "image/png"
			}
			return ImageResult{Bytes: resp.GetImageBytes(), MimeType: mime, ModelVersion: resp.GetModelVersion(), Attempts: attempt}, nil
		}
		lastErr = err
		if !isRetryableImageErr(err) || attempt == i.retry.Attempts {
			break
		}
		delay := i.retry.BaseDelay * time.Duration(1<<(attempt-1))
		if delay > i.retry.MaxDelay {
			delay = i.retry.MaxDelay
		}
		// Full jitter: N renders fan out concurrently, a fixed backoff would
		// re-synchronise them into the same throttled burst.
		delay = time.Duration(float64(delay)*0.5 + rand.Float64()*float64(delay)*0.5)
		if err := i.sleep(ctx, delay); err != nil {
			return ImageResult{Attempts: attempt}, fmt.Errorf("modelgatewayclient: image invoke cancelled: %w (last: %v)", err, lastErr)
		}
	}
	return ImageResult{}, fmt.Errorf("modelgatewayclient: image Invoke: %w", lastErr)
}

// isRetryableImageErr: UNAVAILABLE is the load-bearing one (the gateway wraps
// a vendor HTTP 429 as UNAVAILABLE); the rest are transient by contract.
func isRetryableImageErr(err error) bool {
	switch status.Code(err) {
	case codes.ResourceExhausted, codes.Unavailable, codes.DeadlineExceeded, codes.Aborted, codes.Internal:
		return true
	}
	return false
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// inlineImageContentsJSON renders the genai contents for an edit: one user
// turn with the text prompt and the source image as inlineData (base64).
func inlineImageContentsJSON(prompt string, img []byte, mime string) string {
	if strings.TrimSpace(mime) == "" {
		mime = "image/png"
	}
	contents := []map[string]any{{
		"role": "user",
		"parts": []map[string]any{
			{"text": prompt},
			{"inlineData": map[string]any{"mimeType": mime, "data": base64.StdEncoding.EncodeToString(img)}},
		},
	}}
	b, _ := json.Marshal(contents)
	return string(b)
}
