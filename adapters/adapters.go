package adapters

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"uco-proxy/router"
)

// ProviderAdapter defines the interface for mutating proxy requests into provider-specific payloads.
type ProviderAdapter interface {
	AdaptPayload(ctx context.Context, segments []router.MessageSegment, imageBuffers map[int][]byte, originalPayload []byte) ([]byte, error)
}

// -----------------------------------------------------------------------------
// OpenAI Adapter
// -----------------------------------------------------------------------------

type OpenAIImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type OpenAIContentPart struct {
	Type     string          `json:"type"`
	Text     string          `json:"text,omitempty"`
	ImageURL *OpenAIImageURL `json:"image_url,omitempty"`
}

type OpenAIMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // can be string or []OpenAIContentPart
}

type OpenAIAdapter struct{}

func (a *OpenAIAdapter) AdaptPayload(ctx context.Context, segments []router.MessageSegment, imageBuffers map[int][]byte, originalPayload []byte) ([]byte, error) {
	// Parse original payload to preserve top-level parameters (stream, temp, max_tokens, etc.)
	var rawMap map[string]interface{}
	if err := json.Unmarshal(originalPayload, &rawMap); err != nil {
		return nil, fmt.Errorf("failed to parse original payload: %w", err)
	}

	// 1. Gather all RENDER_BITMAP segments and render them as user image parts
	var imageParts []OpenAIContentPart
	for idx, seg := range segments {
		if seg.Strategy == router.RenderBitmap {
			imgBytes := imageBuffers[idx]
			if len(imgBytes) > 0 {
				b64Data := base64.StdEncoding.EncodeToString(imgBytes)
				imgURL := fmt.Sprintf("data:image/png;base64,%s", b64Data)

				// Prepend a small label giving context of what this image block is
				imageParts = append(imageParts, OpenAIContentPart{
					Type: "text",
					Text: fmt.Sprintf("[Optimized Context Block - Original Role: %s]", seg.Role),
				})
				imageParts = append(imageParts, OpenAIContentPart{
					Type: "image_url",
					ImageURL: &OpenAIImageURL{
						URL:    imgURL,
						Detail: "high",
					},
				})
			}
		}
	}

	// 2. Build new messages list
	var adaptedMessages []OpenAIMessage
	if len(imageParts) > 0 {
		// Multimodal images belong in a user message
		adaptedMessages = append(adaptedMessages, OpenAIMessage{
			Role:    "user",
			Content: imageParts,
		})
	}

	// Append KEEP_TEXT messages
	for _, seg := range segments {
		if seg.Strategy == router.KeepText {
			adaptedMessages = append(adaptedMessages, OpenAIMessage{
				Role:    seg.Role,
				Content: seg.ContentText,
			})
		}
	}

	// Update rawMap with new messages
	rawMap["messages"] = adaptedMessages
	return json.Marshal(rawMap)
}

// -----------------------------------------------------------------------------
// Anthropic Adapter
// -----------------------------------------------------------------------------

type AnthropicImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type AnthropicContentPart struct {
	Type   string                `json:"type"`
	Text   string                `json:"text,omitempty"`
	Source *AnthropicImageSource `json:"source,omitempty"`
}

type AnthropicMessage struct {
	Role    string                 `json:"role"`
	Content []AnthropicContentPart `json:"content"`
}

type AnthropicAdapter struct{}

func (a *AnthropicAdapter) AdaptPayload(ctx context.Context, segments []router.MessageSegment, imageBuffers map[int][]byte, originalPayload []byte) ([]byte, error) {
	// Parse original payload to preserve top-level parameters
	var rawMap map[string]interface{}
	if err := json.Unmarshal(originalPayload, &rawMap); err != nil {
		return nil, fmt.Errorf("failed to parse original payload: %w", err)
	}

	var systemPrompt string
	var imageParts []AnthropicContentPart

	// 1. Gather all RENDER_BITMAP segments and extract standard system prompts
	for idx, seg := range segments {
		// Claude separates the system prompt into a top-level parameter.
		// If system prompt is kept as text, map it there.
		if seg.Role == "system" && seg.Strategy == router.KeepText {
			systemPrompt = seg.ContentText
			continue
		}

		if seg.Strategy == router.RenderBitmap {
			imgBytes := imageBuffers[idx]
			if len(imgBytes) > 0 {
				b64Data := base64.StdEncoding.EncodeToString(imgBytes)

				imageParts = append(imageParts, AnthropicContentPart{
					Type: "text",
					Text: fmt.Sprintf("[Optimized Context Block - Original Role: %s]", seg.Role),
				})
				imageParts = append(imageParts, AnthropicContentPart{
					Type: "image",
					Source: &AnthropicImageSource{
						Type:      "base64",
						MediaType: "image/png",
						Data:      b64Data,
					},
				})
			}
		}
	}

	// 2. Build Claude messages list
	var adaptedMessages []AnthropicMessage
	if len(imageParts) > 0 {
		adaptedMessages = append(adaptedMessages, AnthropicMessage{
			Role:    "user",
			Content: imageParts,
		})
	}

	// Append KEEP_TEXT messages
	for _, seg := range segments {
		if seg.Role == "system" {
			continue // Already processed into System field or as image
		}
		if seg.Strategy == router.KeepText {
			role := seg.Role
			if role == "assistant" {
				role = "assistant"
			} else {
				role = "user"
			}
			adaptedMessages = append(adaptedMessages, AnthropicMessage{
				Role: role,
				Content: []AnthropicContentPart{
					{
						Type: "text",
						Text: seg.ContentText,
					},
				},
			})
		}
	}

	// Update mapping
	rawMap["messages"] = adaptedMessages
	if systemPrompt != "" {
		rawMap["system"] = systemPrompt
	} else {
		delete(rawMap, "system")
	}

	// Ensure max_tokens is present (Claude API requires it)
	if _, exists := rawMap["max_tokens"]; !exists {
		// If max_tokens is not specified, default to a safe value
		rawMap["max_tokens"] = 1000
	}

	return json.Marshal(rawMap)
}

// -----------------------------------------------------------------------------
// Google Gemini Adapter
// -----------------------------------------------------------------------------

type GeminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type GeminiPart struct {
	Text       string            `json:"text,omitempty"`
	InlineData *GeminiInlineData `json:"inlineData,omitempty"`
}

type GeminiContent struct {
	Role  string       `json:"role"`
	Parts []GeminiPart `json:"parts"`
}

type GeminiAdapter struct{}

func (a *GeminiAdapter) AdaptPayload(ctx context.Context, segments []router.MessageSegment, imageBuffers map[int][]byte, originalPayload []byte) ([]byte, error) {
	// Parse original payload to preserve top-level parameters
	var rawMap map[string]interface{}
	if err := json.Unmarshal(originalPayload, &rawMap); err != nil {
		return nil, fmt.Errorf("failed to parse original payload: %w", err)
	}

	var imageParts []GeminiPart
	for idx, seg := range segments {
		if seg.Strategy == router.RenderBitmap {
			imgBytes := imageBuffers[idx]
			if len(imgBytes) > 0 {
				b64Data := base64.StdEncoding.EncodeToString(imgBytes)

				imageParts = append(imageParts, GeminiPart{
					Text: fmt.Sprintf("[Optimized Context Block - Original Role: %s]", seg.Role),
				})
				imageParts = append(imageParts, GeminiPart{
					InlineData: &GeminiInlineData{
						MimeType: "image/png",
						Data:     b64Data,
					},
				})
			}
		}
	}

	// 2. Build Gemini contents
	var adaptedContents []GeminiContent
	if len(imageParts) > 0 {
		adaptedContents = append(adaptedContents, GeminiContent{
			Role:  "user",
			Parts: imageParts,
		})
	}

	// Append KEEP_TEXT messages
	for _, seg := range segments {
		if seg.Strategy == router.KeepText {
			role := seg.Role
			if role == "assistant" {
				role = "model"
			} else {
				role = "user"
			}
			adaptedContents = append(adaptedContents, GeminiContent{
				Role: role,
				Parts: []GeminiPart{
					{
						Text: seg.ContentText,
					},
				},
			})
		}
	}

	// Clean/Update parameters for Gemini
	delete(rawMap, "messages")
	rawMap["contents"] = adaptedContents

	return json.Marshal(rawMap)
}

// -----------------------------------------------------------------------------
// Adapter Factory Loader
// -----------------------------------------------------------------------------

// GetAdapter returns the appropriate ProviderAdapter based on target model/provider.
func GetAdapter(model string) ProviderAdapter {
	m := strings.ToLower(model)
	if strings.Contains(m, "claude") {
		return &AnthropicAdapter{}
	} else if strings.Contains(m, "gemini") {
		return &GeminiAdapter{}
	}
	// Default to OpenAI adapter (standard)
	return &OpenAIAdapter{}
}
