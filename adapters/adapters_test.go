package adapters

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"uco-proxy/router"
)

func TestOpenAIAdapter(t *testing.T) {
	originalPayload := []byte(`{"model": "gpt-4o", "temperature": 0.7, "messages": []}`)
	
	segments := []router.MessageSegment{
		{
			Role:        "system",
			ContentText: "You are a helpful coding assistant.",
			Strategy:    router.RenderBitmap, // Rendered
			IsStatic:    true,
		},
		{
			Role:        "user",
			ContentText: "Please analyze the code.",
			Strategy:    router.KeepText, // Kept
			IsStatic:    false,
		},
	}

	imageBuffers := map[int][]byte{
		0: []byte("fake-png-bytes-for-system-prompt"),
	}

	adapter := &OpenAIAdapter{}
	mutatedBytes, err := adapter.AdaptPayload(context.Background(), segments, imageBuffers, originalPayload)
	if err != nil {
		t.Fatalf("Failed to adapt payload for OpenAI: %v", err)
	}

	// Unmarshal mutated payload
	var mutatedMap map[string]interface{}
	if err := json.Unmarshal(mutatedBytes, &mutatedMap); err != nil {
		t.Fatalf("Mutated payload is invalid JSON: %v", err)
	}

	// Verify temperature is preserved
	if mutatedMap["temperature"].(float64) != 0.7 {
		t.Errorf("Expected temperature 0.7, got %v", mutatedMap["temperature"])
	}

	// Verify messages structure
	messagesRaw, exists := mutatedMap["messages"]
	if !exists {
		t.Fatal("Mutated payload missing messages array")
	}

	messagesBytes, _ := json.Marshal(messagesRaw)
	var messages []OpenAIMessage
	_ = json.Unmarshal(messagesBytes, &messages)

	if len(messages) != 2 {
		t.Fatalf("Expected 2 messages in adapted payload, got %d", len(messages))
	}

	// First message should be multimodal user message (our consolidated image blocks)
	firstMsg := messages[0]
	if firstMsg.Role != "user" {
		t.Errorf("Expected first message role to be user, got %s", firstMsg.Role)
	}

	// Content should be array of parts
	partsList, ok := firstMsg.Content.([]interface{})
	if !ok {
		t.Fatalf("Expected first message content to be parts slice, got %T", firstMsg.Content)
	}

	hasImage := false
	for _, partRaw := range partsList {
		partMap, ok := partRaw.(map[string]interface{})
		if !ok {
			continue
		}
		if partMap["type"] == "image_url" {
			hasImage = true
			imgURLMap := partMap["image_url"].(map[string]interface{})
			urlStr := imgURLMap["url"].(string)
			if !strings.HasPrefix(urlStr, "data:image/png;base64,") {
				t.Errorf("Expected base64 PNG prefix in image url, got %s", urlStr)
			}
		}
	}

	if !hasImage {
		t.Error("Expected first message parts to contain an image block")
	}

	// Second message should be standard text user query
	secondMsg := messages[1]
	if secondMsg.Role != "user" {
		t.Errorf("Expected second message role to be user, got %s", secondMsg.Role)
	}
	if secondMsg.Content.(string) != "Please analyze the code." {
		t.Errorf("Expected content 'Please analyze the code.', got %v", secondMsg.Content)
	}
}

func TestAnthropicAdapter(t *testing.T) {
	originalPayload := []byte(`{"model": "claude-3-5-sonnet-20241022"}`)
	
	segments := []router.MessageSegment{
		{
			Role:        "system",
			ContentText: "System prompt text.",
			Strategy:    router.KeepText, // System prompt kept as text
			IsStatic:    true,
		},
		{
			Role:        "user",
			ContentText: "Huge file content here...",
			Strategy:    router.RenderBitmap, // Large block optimized
			IsStatic:    true,
		},
		{
			Role:        "user",
			ContentText: "Find the bug.",
			Strategy:    router.KeepText, // Dynamic query
			IsStatic:    false,
		},
	}

	imageBuffers := map[int][]byte{
		1: []byte("fake-png-bytes-for-history"),
	}

	adapter := &AnthropicAdapter{}
	mutatedBytes, err := adapter.AdaptPayload(context.Background(), segments, imageBuffers, originalPayload)
	if err != nil {
		t.Fatalf("Failed to adapt payload for Anthropic: %v", err)
	}

	var mutatedMap map[string]interface{}
	_ = json.Unmarshal(mutatedBytes, &mutatedMap)

	// Verify system prompt parameter
	sysParam, exists := mutatedMap["system"]
	if !exists {
		t.Error("Expected top-level system parameter for Claude, got none")
	} else if sysParam.(string) != "System prompt text." {
		t.Errorf("Expected system parameter 'System prompt text.', got %s", sysParam)
	}

	// Verify messages
	messagesRaw := mutatedMap["messages"]
	messagesBytes, _ := json.Marshal(messagesRaw)
	var messages []AnthropicMessage
	_ = json.Unmarshal(messagesBytes, &messages)

	if len(messages) != 2 {
		t.Fatalf("Expected 2 messages, got %d", len(messages))
	}

	// First message contains image block
	firstMsg := messages[0]
	if firstMsg.Role != "user" {
		t.Errorf("Expected first message role user, got %s", firstMsg.Role)
	}
	hasImage := false
	for _, part := range firstMsg.Content {
		if part.Type == "image" {
			hasImage = true
			if part.Source.Type != "base64" || part.Source.MediaType != "image/png" {
				t.Errorf("Expected base64 image/png source, got %+v", part.Source)
			}
		}
	}
	if !hasImage {
		t.Error("Expected image block in Claude user message content")
	}

	// Second message contains dynamic user query
	secondMsg := messages[1]
	if secondMsg.Role != "user" {
		t.Errorf("Expected second message role user, got %s", secondMsg.Role)
	}
	if secondMsg.Content[0].Text != "Find the bug." {
		t.Errorf("Expected 'Find the bug.', got %s", secondMsg.Content[0].Text)
	}
}

func TestGeminiAdapter(t *testing.T) {
	originalPayload := []byte(`{"model": "gemini-1.5-pro"}`)
	
	segments := []router.MessageSegment{
		{
			Role:        "system",
			ContentText: "System prompt instructions.",
			Strategy:    router.RenderBitmap, // Optimized system prompt
			IsStatic:    true,
		},
		{
			Role:        "user",
			ContentText: "Write a function.",
			Strategy:    router.KeepText,
			IsStatic:    false,
		},
	}

	imageBuffers := map[int][]byte{
		0: []byte("fake-png-bytes-for-system"),
	}

	adapter := &GeminiAdapter{}
	mutatedBytes, err := adapter.AdaptPayload(context.Background(), segments, imageBuffers, originalPayload)
	if err != nil {
		t.Fatalf("Failed to adapt payload for Gemini: %v", err)
	}

	var mutatedMap map[string]interface{}
	_ = json.Unmarshal(mutatedBytes, &mutatedMap)

	// Verify messages key is deleted and contents key is added
	if _, exists := mutatedMap["messages"]; exists {
		t.Error("Expected messages key to be deleted for Gemini payload")
	}

	contentsRaw, exists := mutatedMap["contents"]
	if !exists {
		t.Fatal("Missing contents key for Gemini payload")
	}

	contentsBytes, _ := json.Marshal(contentsRaw)
	var contents []GeminiContent
	_ = json.Unmarshal(contentsBytes, &contents)

	if len(contents) != 2 {
		t.Fatalf("Expected 2 content structures, got %d", len(contents))
	}

	// Verify first message contains inlineData
	firstMsg := contents[0]
	if firstMsg.Role != "user" {
		t.Errorf("Expected first content role user, got %s", firstMsg.Role)
	}
	hasImage := false
	for _, part := range firstMsg.Parts {
		if part.InlineData != nil {
			hasImage = true
			if part.InlineData.MimeType != "image/png" {
				t.Errorf("Expected image/png, got %s", part.InlineData.MimeType)
			}
		}
	}
	if !hasImage {
		t.Error("Expected inlineData image block in Gemini parts list")
	}

	// Verify second message contains dynamic text query
	secondMsg := contents[1]
	if secondMsg.Role != "user" {
		t.Errorf("Expected second content role user, got %s", secondMsg.Role)
	}
	if secondMsg.Parts[0].Text != "Write a function." {
		t.Errorf("Expected text 'Write a function.', got %s", secondMsg.Parts[0].Text)
	}
}
