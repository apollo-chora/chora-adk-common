// Package groundingplugin — EPIC-1a batch source-material grounding.
//
// A BeforeModelCallback that injects the uploaded batch source material as a
// genai FileData (object-store URI by reference) part into the agent's LLM request, so the
// model grounds generation on the document. The orchestrator threads
// `source_blob_uri` + `source_mime_type` into ADK session state
// (reasoning_engine_executor._build_session_state); this plugin reads them and
// appends a FileData part to the last user content. modelgatewayclient carries
// the part as contents_json → the model gateway reads the object-store URI DIRECTLY (no
// gateway download; no gRPC-4MB inline-base64 blowout on a large PDF).
//
// Empty source_blob_uri ⇒ no-op: the live single-candidate + non-grounded batch
// paths are byte-for-byte unchanged. Register alongside the tenant-propagation
// plugin in the launcher PluginConfig for any crew that grounds on uploaded
// material (the EPIC-1a qgen batch path).
package groundingplugin

import (
	"strings"

	"google.golang.org/adk/agent"
	adkmodel "google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/genai"
)

// Session-state keys the orchestrator threads (generate_node →
// _build_session_state). Match the chora-contracts batch field names.
const (
	stateKeySourceBlobURI  = "source_blob_uri"
	stateKeySourceMimeType = "source_mime_type"
)

// stateReader is the minimal session-state read surface (matches
// session.ReadonlyState / session.State Get). Mirrors manaplugin.getStateString.
type stateReader interface {
	Get(key string) (any, error)
}

func readState(state stateReader, key string) string {
	v, err := state.Get(key)
	if err != nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

// applyGroundingFromState appends an object-store URI FileData part to the last user content
// of req when session state carries a source_blob_uri. Factored out for unit
// testing without a full ADK CallbackContext. No-op on empty URI / nil req.
func applyGroundingFromState(state stateReader, req *adkmodel.LLMRequest) {
	if req == nil {
		return
	}
	blobURI := strings.TrimSpace(readState(state, stateKeySourceBlobURI))
	if blobURI == "" {
		return // no grounding material — single + non-grounded paths unchanged
	}
	mime := strings.TrimSpace(readState(state, stateKeySourceMimeType))
	part := &genai.Part{FileData: &genai.FileData{FileURI: blobURI, MIMEType: mime}}

	// Append to the LAST user content (the turn-initiating message) so the blob
	// rides with the user's instruction. If there is no user content (defensive)
	// append a fresh user content carrying the part.
	for i := len(req.Contents) - 1; i >= 0; i-- {
		c := req.Contents[i]
		if c == nil {
			continue
		}
		if c.Role == "user" || c.Role == "" {
			c.Parts = append(c.Parts, part)
			return
		}
	}
	req.Contents = append(req.Contents, &genai.Content{Role: "user", Parts: []*genai.Part{part}})
}

// New returns an ADK plugin whose BeforeModelCallback injects the batch source
// material (source_blob_uri from session state) as an object-store URI FileData part into the
// LLM request. Wire alongside the tenant-propagation plugin in the launcher
// PluginConfig for any crew that grounds on uploaded material (EPIC-1a qgen).
func New(crewKind string) (*plugin.Plugin, error) {
	return plugin.New(plugin.Config{
		Name: "chora_grounding_" + crewKind,
		BeforeModelCallback: func(ctx agent.CallbackContext, req *adkmodel.LLMRequest) (*adkmodel.LLMResponse, error) {
			applyGroundingFromState(ctx.ReadonlyState(), req)
			return nil, nil
		},
	})
}
