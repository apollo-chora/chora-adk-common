// Package groundingplugin — EPIC-1a batch source-material grounding.
//
// A BeforeModelCallback that injects the uploaded batch source material into
// the agent's LLM request so the model grounds generation on the document.
//
// # THE MATERIAL IS TEXT, NOT A BLOB REFERENCE
//
// This plugin used to append a genai FileData part carrying the object-store
// URI "by reference", on the assumption that the gateway would dereference it.
// That assumption died with the Go gateway: the Python model gateway's
// canonical contract admits only text | image | refusal parts, so
// modelgatewayclient refuses a FileData part outright and every file-based
// generation failed before a model call.
//
// The material now rides as ordinary text. The orchestrator fetches the job's
// extracted, page-referenced chunks from chora-creation
// (GET /api/internal/question-jobs/{job_id}/source-material), renders them
// with explicit file/page delimiters, and stamps the result into session state
// under `source_material_text`. This plugin injects that string verbatim: it
// does NOT reconstruct document semantics, because it cannot see the chunk
// structure — only the orchestrator can.
//
// # FAIL LOUD, NEVER GROUND SILENTLY
//
// If the state declares source material but carries no text, this plugin
// returns an error rather than proceeding ungrounded. Generating unrelated
// questions while the author believes they came from their document is worse
// than any error message. Image-only uploads land here too: they have no text
// layer by design (ADR-180 D9), so they cannot be text-grounded until a
// multimodal path exists.
//
// No source material declared ⇒ no-op, so the live single-candidate and
// non-grounded batch paths stay byte-for-byte unchanged. Register alongside the
// tenant-propagation plugin in the launcher PluginConfig for any crew that
// grounds on uploaded material (the EPIC-1a qgen batch path).
package groundingplugin

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"google.golang.org/adk/agent"
	adkmodel "google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/genai"
)

// Session-state keys the orchestrator threads (generate_node →
// _build_session_state). Match the chora-contracts batch field names.
const (
	stateKeySourceBlobURI   = "source_blob_uri"
	stateKeySourceMimeType  = "source_mime_type"
	stateKeySourceFilesJSON = "source_files_json"

	// stateKeySourceMaterialText carries the ALREADY-RENDERED, delimited
	// grounding text. Presence of this key is what makes grounding happen;
	// the three keys above are only a declaration that material exists.
	stateKeySourceMaterialText = "source_material_text"
)

// ErrNoSourceMaterialText is returned when the job declares uploaded source
// material but no grounding text arrived. Callers see a failed model call with
// this reason rather than an ungrounded generation.
var ErrNoSourceMaterialText = errors.New(
	"groundingplugin: job declares source material but source_material_text is empty — " +
		"refusing to generate ungrounded (image-only uploads and failed extractions have no text layer)",
)

// sourceFile is one entry of the role-tagged effective file list the
// orchestrator forwards for a multi-file (or rubric-bearing) grounded batch.
// Only its PRESENCE matters here now — the text itself comes from
// source_material_text.
type sourceFile struct {
	BlobURI  string `json:"blob_uri"`
	MimeType string `json:"mime_type"`
	Role     string `json:"role"`
}

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

// declaresSourceMaterial reports whether the orchestrator said this job has
// uploaded material, regardless of whether any text arrived.
func declaresSourceMaterial(state stateReader) bool {
	if strings.TrimSpace(readState(state, stateKeySourceMaterialText)) != "" {
		return true
	}
	if raw := strings.TrimSpace(readState(state, stateKeySourceFilesJSON)); raw != "" {
		var files []sourceFile
		if err := json.Unmarshal([]byte(raw), &files); err == nil {
			for _, f := range files {
				if strings.TrimSpace(f.BlobURI) != "" {
					return true
				}
			}
		}
	}
	return strings.TrimSpace(readState(state, stateKeySourceBlobURI)) != ""
}

// applyGroundingFromState appends the job's grounding TEXT to the last user
// content of req. Factored out for unit testing without a full ADK
// CallbackContext.
//
// Returns ErrNoSourceMaterialText when material is declared but no text is
// present, and nil for a job with no material at all (the unchanged paths).
func applyGroundingFromState(state stateReader, req *adkmodel.LLMRequest) error {
	if req == nil {
		return nil
	}

	text := strings.TrimSpace(readState(state, stateKeySourceMaterialText))
	if text == "" {
		if declaresSourceMaterial(state) {
			return ErrNoSourceMaterialText
		}
		return nil // no grounding material — single + non-grounded paths unchanged
	}

	// A leading separator keeps the grounding block from gluing onto the
	// author's instruction when the constructor did not end with a newline.
	appendParts(req, []*genai.Part{{Text: "\n\n" + text}})
	return nil
}

// appendParts rides the parts on the LAST user content (the turn-initiating
// message). If there is no user content (defensive) it appends a fresh one.
func appendParts(req *adkmodel.LLMRequest, parts []*genai.Part) {
	for i := len(req.Contents) - 1; i >= 0; i-- {
		c := req.Contents[i]
		if c == nil {
			continue
		}
		if c.Role == "user" || c.Role == "" {
			c.Parts = append(c.Parts, parts...)
			return
		}
	}
	req.Contents = append(req.Contents, &genai.Content{Role: "user", Parts: parts})
}

// New returns an ADK plugin whose BeforeModelCallback injects the job's
// grounding text into the LLM request. Wire alongside the tenant-propagation
// plugin in the launcher PluginConfig for any crew that grounds on uploaded
// material (EPIC-1a qgen).
func New(crewKind string) (*plugin.Plugin, error) {
	return plugin.New(plugin.Config{
		Name: "chora_grounding_" + crewKind,
		BeforeModelCallback: func(ctx agent.CallbackContext, req *adkmodel.LLMRequest) (*adkmodel.LLMResponse, error) {
			if err := applyGroundingFromState(ctx.ReadonlyState(), req); err != nil {
				return nil, fmt.Errorf("chora_grounding_%s: %w", crewKind, err)
			}
			return nil, nil
		},
	})
}
