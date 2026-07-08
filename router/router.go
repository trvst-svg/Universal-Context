package router

import (
	"math"
	"strings"

	"github.com/pkoukk/tiktoken-go"
)

type OptimizationStrategy string

const (
	KeepText     OptimizationStrategy = "KEEP_TEXT"
	RenderBitmap OptimizationStrategy = "RENDER_BITMAP"
)

// Message represents an incoming OpenAI-format chat completion message.
type Message struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // can be string or array of blocks
}

// GetContentText extracts the plain text from the message content.
// Handles both plain strings and arrays of content blocks (for multimodal payloads).
func (m Message) GetContentText() string {
	if m.Content == nil {
		return ""
	}
	if str, ok := m.Content.(string); ok {
		return str
	}
	// Handle array of blocks (common in multimodal payloads)
	if blocks, ok := m.Content.([]interface{}); ok {
		var textParts []string
		for _, b := range blocks {
			if blockMap, ok := b.(map[string]interface{}); ok {
				if blockMap["type"] == "text" {
					if text, ok := blockMap["text"].(string); ok {
						textParts = append(textParts, text)
					}
				}
			}
		}
		return strings.Join(textParts, "\n")
	}
	return ""
}

// MessageSegment represents our internal decorated segment with budgeting analysis.
type MessageSegment struct {
	Role                  string               `json:"role"`
	ContentText           string               `json:"content_text"`
	TextTokens            int                  `json:"text_tokens"`
	EstimatedVisionTokens int                  `json:"estimated_vision_tokens"`
	Strategy              OptimizationStrategy `json:"strategy"`
	IsStatic              bool                 `json:"is_static"`
}

// PayloadAnalysis summarizes the routing decisions for the entire payload.
type PayloadAnalysis struct {
	Model    string           `json:"model"`
	Segments []MessageSegment `json:"segments"`
}

// ChatPayload represents the minimal incoming chat completion body for parsing.
type ChatPayload struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

// TokenCounter defines the signature for counting text tokens.
type TokenCounter func(text string, model string) int

// DefaultTokenCounter is the default implementation using tiktoken-go.
// If tiktoken-go fails to download BPE assets, it falls back to a character count heuristic (char_len / 4).
var DefaultTokenCounter TokenCounter = func(text string, model string) int {
	// Clean model name for lookup
	encodingModel := "gpt-4o"
	if strings.Contains(model, "claude") {
		// Use gpt-4o tiktoken model as a proxy for Claude text tokens since Claude
		// token count is very close to cl100k/o200k base.
		encodingModel = "gpt-4o"
	} else if model != "" {
		encodingModel = model
	}

	tkm, err := tiktoken.EncodingForModel(encodingModel)
	if err != nil {
		// Fallback to standard base encoding
		tkm, err = tiktoken.GetEncoding("cl100k_base")
		if err != nil {
			// Local offline fallback
			return int(math.Ceil(float64(len(text)) / 4.0))
		}
	}

	return len(tkm.Encode(text, nil, nil))
}

// CalculateOpenAIVisionTokens computes the high-detail vision tokens for a given image size.
// Following OpenAI's formula: 85 base + 170 per 512x512 tile after resizing.
func CalculateOpenAIVisionTokens(width, height int) int {
	w, h := float64(width), float64(height)

	// Step 1: Scale shortest side to 768px
	if w < h {
		scale := 768.0 / w
		w = 768
		h = math.Round(h * scale)
	} else {
		scale := 768.0 / h
		h = 768
		w = math.Round(w * scale)
	}

	// Step 2: Scale down to fit 2048x2048 if necessary
	if w > 2048 || h > 2048 {
		scale := math.Min(2048/w, 2048/h)
		w = math.Round(w * scale)
		h = math.Round(h * scale)
	}

	// Step 3: Count 512x512 tiles
	tilesX := math.Ceil(w / 512.0)
	tilesY := math.Ceil(h / 512.0)

	// Step 4: Token cost is 85 base + 170 per tile
	return 85 + 170*int(tilesX*tilesY)
}

// wrapText wraps text at word boundaries so that no line exceeds maxChars.
func wrapText(text string, maxChars int) ([]string, int) {
	text = strings.ReplaceAll(text, "\t", "    ")
	var wrapped []string
	lines := strings.Split(text, "\n")
	maxLineLen := 0

	for _, line := range lines {
		if len(line) == 0 {
			wrapped = append(wrapped, "")
			continue
		}

		runes := []rune(line)
		for len(runes) > 0 {
			if len(runes) <= maxChars {
				wrapped = append(wrapped, string(runes))
				if len(runes) > maxLineLen {
					maxLineLen = len(runes)
				}
				break
			}

			// Look for space to wrap at word boundary
			wrapIdx := maxChars
			for i := maxChars; i > 0; i-- {
				if runes[i] == ' ' {
					wrapIdx = i
					break
				}
			}

			// Wrap at wrapIdx
			segment := runes[:wrapIdx]
			wrapped = append(wrapped, string(segment))
			if len(segment) > maxLineLen {
				maxLineLen = len(segment)
			}

			// Remainder
			runes = runes[wrapIdx:]
			// Strip leading space of remainder
			if len(runes) > 0 && runes[0] == ' ' {
				runes = runes[1:]
			}
		}
	}
	return wrapped, maxLineLen
}

// EstimateTextDimensions calculates layout dimensions for a text block.
// It applies text wrapping at 80 characters to match the custom renderer's layout behavior.
func EstimateTextDimensions(text string, charWidth, lineHeight, margin int) (int, int) {
	lines, maxLineLen := wrapText(text, 80)
	numLines := len(lines)

	width := maxLineLen*charWidth + margin*2
	height := numLines*lineHeight + margin*2

	// Set safe minimum bounds
	if width < 100 {
		width = 100
	}
	if height < 100 {
		height = 100
	}

	return width, height
}

// AnalyzePayload evaluates the incoming request messages and maps optimization strategies.
func AnalyzePayload(payload ChatPayload, counter TokenCounter) PayloadAnalysis {
	analysis := PayloadAnalysis{
		Model:    payload.Model,
		Segments: make([]MessageSegment, 0, len(payload.Messages)),
	}

	numMessages := len(payload.Messages)
	if numMessages == 0 {
		return analysis
	}

	for idx, msg := range payload.Messages {
		text := msg.GetContentText()

		// The last message in the array represents the active User Instruction (Dynamic Context)
		// Everything else (system prompt, older assistant/user history) is Static Context
		isStatic := idx < (numMessages - 1)

		var strategy OptimizationStrategy
		var textTokens int
		var visionTokens int

		if !isStatic {
			// Dynamic Context is ALWAYS kept as text
			strategy = KeepText
			textTokens = counter(text, payload.Model)
			visionTokens = 0
		} else {
			// Static Context runs through the Token Budgeting decision engine
			textTokens = counter(text, payload.Model)

			// Estimate rendered dimensions (Custom 10x16 bitmap metrics)
			w, h := EstimateTextDimensions(text, 10, 16, 20)
			visionTokens = CalculateOpenAIVisionTokens(w, h)

			// Budget decision: If vision tile representation is cheaper, render it
			if textTokens > visionTokens {
				strategy = RenderBitmap
			} else {
				strategy = KeepText
			}
		}

		analysis.Segments = append(analysis.Segments, MessageSegment{
			Role:                  msg.Role,
			ContentText:           text,
			TextTokens:            textTokens,
			EstimatedVisionTokens: visionTokens,
			Strategy:              strategy,
			IsStatic:              isStatic,
		})
	}

	return analysis
}
