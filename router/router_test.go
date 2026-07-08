package router

import (
	"testing"
)

func TestAnalyzePayload(t *testing.T) {
	// Mock token counter to run test cases deterministically without relying on network-based BPE downloads
	mockCounterMap := map[string]int{
		"small_system": 10,
		"large_code":   3500,
		"user_query":   15,
		"large_query":  5000,
	}

	mockCounter := func(text string, model string) int {
		if val, exists := mockCounterMap[text]; exists {
			return val
		}
		return len(text) / 4 // Simple fallback
	}

	tests := []struct {
		name             string
		payload          ChatPayload
		expectedRoles    []string
		expectedStatics  []bool
		expectedStrategy []OptimizationStrategy
	}{
		{
			name: "Small system and small user prompt - both keep text",
			payload: ChatPayload{
				Model: "gpt-4o",
				Messages: []Message{
					{Role: "system", Content: "small_system"},
					{Role: "user", Content: "user_query"},
				},
			},
			expectedRoles:    []string{"system", "user"},
			expectedStatics:  []bool{true, false},
			expectedStrategy: []OptimizationStrategy{KeepText, KeepText},
		},
		{
			name: "Large static context and small user query - static should render bitmap, user keep text",
			payload: ChatPayload{
				Model: "gpt-4o",
				Messages: []Message{
					{Role: "system", Content: "large_code"}, // 3500 mock tokens
					{Role: "user", Content: "user_query"},
				},
			},
			expectedRoles:    []string{"system", "user"},
			expectedStatics:  []bool{true, false},
			expectedStrategy: []OptimizationStrategy{RenderBitmap, KeepText},
		},
		{
			name: "Large user prompt at the end (dynamic context) - must always keep text",
			payload: ChatPayload{
				Model: "gpt-4o",
				Messages: []Message{
					{Role: "system", Content: "small_system"},
					{Role: "user", Content: "large_query"}, // 5000 mock tokens (should remain text)
				},
			},
			expectedRoles:    []string{"system", "user"},
			expectedStatics:  []bool{true, false},
			expectedStrategy: []OptimizationStrategy{KeepText, KeepText},
		},
		{
			name: "Mixed context history",
			payload: ChatPayload{
				Model: "gpt-4o",
				Messages: []Message{
					{Role: "system", Content: "small_system"}, // KeepText (short)
					{Role: "user", Content: "large_code"},     // RenderBitmap (large static history)
					{Role: "assistant", Content: "small_system"}, // KeepText (short)
					{Role: "user", Content: "user_query"},     // KeepText (dynamic query)
				},
			},
			expectedRoles:    []string{"system", "user", "assistant", "user"},
			expectedStatics:  []bool{true, true, true, false},
			expectedStrategy: []OptimizationStrategy{KeepText, RenderBitmap, KeepText, KeepText},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			analysis := AnalyzePayload(tt.payload, mockCounter)

			if len(analysis.Segments) != len(tt.expectedRoles) {
				t.Fatalf("Expected %d segments, got %d", len(tt.expectedRoles), len(analysis.Segments))
			}

			for idx, seg := range analysis.Segments {
				if seg.Role != tt.expectedRoles[idx] {
					t.Errorf("Segment %d: expected role %s, got %s", idx, tt.expectedRoles[idx], seg.Role)
				}
				if seg.IsStatic != tt.expectedStatics[idx] {
					t.Errorf("Segment %d: expected IsStatic %t, got %t", idx, tt.expectedStatics[idx], seg.IsStatic)
				}
				if seg.Strategy != tt.expectedStrategy[idx] {
					t.Errorf("Segment %d: expected Strategy %s, got %s (TextTokens: %d, VisionTokens: %d)",
						idx, tt.expectedStrategy[idx], seg.Strategy, seg.TextTokens, seg.EstimatedVisionTokens)
				}
			}
		})
	}
}

func TestCalculateOpenAIVisionTokens(t *testing.T) {
	// Standard test case from OpenAI docs:
	// A 1024x1024 image should cost 765 tokens.
	tokens := CalculateOpenAIVisionTokens(1024, 1024)
	if tokens != 765 {
		t.Errorf("Expected 765 tokens for 1024x1024 image, got %d", tokens)
	}

	// 512x512 image should scale shortest side (512) to 768 -> becomes 768x768 (4 tiles) -> 765 tokens.
	tokensShort := CalculateOpenAIVisionTokens(512, 512)
	if tokensShort != 765 {
		t.Errorf("Expected 765 tokens for 512x512 image, got %d", tokensShort)
	}

	// 1024x4096 image has a > 2.67 aspect ratio. Shortest side scales to 768 -> 768x3072.
	// Longest side (3072) is > 2048, so it scales down to 2048 -> 512x2048.
	// Tiles = ceil(512/512)*ceil(2048/512) = 1*4 = 4 tiles -> 765 tokens.
	tokensAspect := CalculateOpenAIVisionTokens(1024, 4096)
	if tokensAspect != 765 {
		t.Errorf("Expected 765 tokens for 1024x4096 aspect ratio, got %d", tokensAspect)
	}
}
