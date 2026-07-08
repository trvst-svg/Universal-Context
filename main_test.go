package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

func TestProxyIntegration(t *testing.T) {
	// 1. Spin up mock upstream OpenAI server
	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify auth bearer was overridden to standard upstream key
		authHeader := r.Header.Get("Authorization")
		if authHeader != "Bearer mock-openai-key" {
			t.Errorf("Mock Upstream: Expected Authorization Bearer mock-openai-key, got: %s", authHeader)
			w.WriteHeader(http.StatusUnauthorized)
			return
		}

		// Read request body to verify adapter payload mutation
		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("Mock Upstream failed to read body: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		// Verify body is mutated into multimodal format containing Base64 image
		bodyStr := string(bodyBytes)
		if !strings.Contains(bodyStr, "data:image/png;base64,") {
			t.Errorf("Mock Upstream body did not contain image base64 data: %s", bodyStr)
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		// Verify the original raw text of system prompt is removed or changed
		if strings.Contains(bodyStr, "\"role\":\"system\"") && strings.Contains(bodyStr, "This is dense, token-heavy documentation") {
			t.Error("Mock Upstream body still contained raw static text under original role that should have been optimized")
		}

		// Return mock SSE stream chunks
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)

		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\" world!\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer mockUpstream.Close()

	// Find a free TCP port dynamically
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Failed to find free port: %v", err)
	}
	freePort := fmt.Sprintf("%d", ln.Addr().(*net.TCPAddr).Port)
	ln.Close()

	// 2. Set environment variables for testing configuration
	os.Setenv("UPSTREAM_URL", mockUpstream.URL)
	os.Setenv("PORT", freePort)
	os.Setenv("DB_PATH", ":memory:")
	os.Setenv("OPENAI_API_KEY", "mock-openai-key")

	// 3. Start main server in background
	go main()

	// Wait actively for the proxy server to start listening
	serverAddr := "127.0.0.1:" + freePort
	started := false
	for i := 0; i < 40; i++ {
		conn, err := net.DialTimeout("tcp", serverAddr, 100*time.Millisecond)
		if err == nil {
			conn.Close()
			started = true
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	if !started {
		t.Fatalf("Server failed to start listening on %s within timeout", serverAddr)
	}

	client := &http.Client{}

	// Test case A: Request without API key (expects 401)
	reqA, _ := http.NewRequest("POST", "http://localhost:"+freePort+"/v1/chat/completions", bytes.NewBuffer([]byte(`{}`)))
	respA, err := client.Do(reqA)
	if err != nil {
		t.Fatalf("Test case A request failed: %v", err)
	}
	if respA.StatusCode != http.StatusUnauthorized {
		t.Errorf("Test case A: Expected 401, got %d", respA.StatusCode)
	}
	respA.Body.Close()

	// Test case B: Request with invalid API key (expects 401)
	reqB, _ := http.NewRequest("POST", "http://localhost:"+freePort+"/v1/chat/completions", bytes.NewBuffer([]byte(`{}`)))
	reqB.Header.Set("X-UCO-API-Key", "invalid-key")
	respB, err := client.Do(reqB)
	if err != nil {
		t.Fatalf("Test case B request failed: %v", err)
	}
	if respB.StatusCode != http.StatusUnauthorized {
		t.Errorf("Test case B: Expected 401, got %d", respB.StatusCode)
	}
	respB.Body.Close()

	// Generate 300 lines of system prompt to trigger RENDER_BITMAP optimization strategy
	largeLines := make([]string, 300)
	for i := 0; i < 300; i++ {
		largeLines[i] = fmt.Sprintf("Line %03d: This is dense, token-heavy documentation describing system behaviors and codebase APIs.", i)
	}
	largeSystemPrompt := strings.Join(largeLines, "\n")

	payloadMap := map[string]interface{}{
		"model": "gpt-4o",
		"messages": []map[string]interface{}{
			{"role": "system", "content": largeSystemPrompt},
			{"role": "user", "content": "What is the summary of the above?"},
		},
		"stream": true,
	}
	payloadBytes, _ := json.Marshal(payloadMap)

	// Test case C: Valid Request with seeded key (expects 200 and streaming chunks)
	reqC, _ := http.NewRequest("POST", "http://localhost:"+freePort+"/v1/chat/completions", bytes.NewBuffer(payloadBytes))
	reqC.Header.Set("Authorization", "Bearer uco-test-key-12345") // Using Bearer auth key style
	respC, err := client.Do(reqC)
	if err != nil {
		t.Fatalf("Test case C request failed: %v", err)
	}
	if respC.StatusCode != http.StatusOK {
		t.Errorf("Test case C: Expected 200, got %d", respC.StatusCode)
	}

	// Read and verify streaming chunks (SSE)
	reader := bufio.NewReader(respC.Body)
	chunksCount := 0
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("Failed to read chunk: %v", err)
		}
		if strings.HasPrefix(line, "data: ") {
			chunksCount++
		}
	}
	respC.Body.Close()

	if chunksCount < 2 {
		t.Errorf("Expected at least 2 data chunks, got %d", chunksCount)
	}

	// Query health check route to verify health states
	reqH, _ := http.NewRequest("GET", "http://localhost:"+freePort+"/health", nil)
	respH, err := client.Do(reqH)
	if err != nil {
		t.Fatalf("Health check request failed: %v", err)
	}
	if respH.StatusCode != http.StatusOK {
		t.Errorf("Health check: Expected 200, got %d", respH.StatusCode)
	}

	var health map[string]interface{}
	_ = json.NewDecoder(respH.Body).Decode(&health)
	respH.Body.Close()

	if health["status"] != "healthy" {
		t.Errorf("Expected healthy status, got %v", health["status"])
	}
	if health["database"] != "online" {
		t.Errorf("Expected database online, got %v", health["database"])
	}
}
