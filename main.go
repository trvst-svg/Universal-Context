package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"

	"uco-proxy/router"
)

func main() {
	targetURL, err := url.Parse("https://api.openai.com")
	if err != nil {
		log.Fatalf("Error parsing target URL: %v", err)
	}

	// Configure ReverseProxy with immediate flush (-1)
	proxy := &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			req.URL.Scheme = targetURL.Scheme
			req.URL.Host = targetURL.Host
			req.Host = targetURL.Host
			// Ensure connection headers are handled properly by proxying
			if clientIP := req.Header.Get("X-Forwarded-For"); clientIP != "" {
				req.Header.Set("X-Forwarded-For", clientIP+", "+req.RemoteAddr)
			} else {
				req.Header.Set("X-Forwarded-For", req.RemoteAddr)
			}
		},
		// FlushInterval of -1 ensures Go flushes response body chunks to the client
		// immediately, which is crucial for SSE (streaming) responsiveness.
		FlushInterval: -1,
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			log.Printf("[UCO Error] Upstream connection failure: %v", err)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			_, _ = w.Write([]byte(`{"error": {"message": "UCO Proxy: Gateway connection to upstream failed."}}`))
		},
	}

	// Setup Server Multiplexer
	mux := http.NewServeMux()

	// Intercept OpenAI chat completions route
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusMethodNotAllowed)
			_, _ = w.Write([]byte(`{"error": {"message": "Only POST requests are supported on /v1/chat/completions"}}`))
			return
		}

		// Extract and mask Authorization header token for validation logging
		auth := r.Header.Get("Authorization")
		var masked string
		if strings.HasPrefix(auth, "Bearer ") {
			token := strings.TrimPrefix(auth, "Bearer ")
			if len(token) > 8 {
				masked = token[:4] + "..." + token[len(token)-4:]
			} else if len(token) > 0 {
				masked = "..."
			} else {
				masked = "<empty>"
			}
		} else {
			masked = "<missing/invalid>"
		}
		log.Printf("[UCO Info] Intercepted POST /v1/chat/completions. Token: %s", masked)

		// Read the request body bytes
		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			log.Printf("[UCO Error] Failed to read request body: %v", err)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error": {"message": "Failed to read request body."}}`))
			return
		}

		// Restore request body for downstream reverse proxy forwarding
		r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

		// Parse chat payload for ContextRouter analysis
		var payload router.ChatPayload
		if err := json.Unmarshal(bodyBytes, &payload); err != nil {
			log.Printf("[UCO Warning] Failed to parse request JSON: %v. Forwarding raw payload.", err)
		} else {
			// Run routing analysis using token budgeting
			analysis := router.AnalyzePayload(payload, router.DefaultTokenCounter)
			log.Printf("[UCO Info] ContextRouter Analysis for model '%s':", analysis.Model)
			for idx, seg := range analysis.Segments {
				log.Printf("  - Msg %d [%s] (Static: %t) -> Strategy: %s (Text: %d T, Vision Est: %d T)",
					idx, seg.Role, seg.IsStatic, seg.Strategy, seg.TextTokens, seg.EstimatedVisionTokens)
			}
		}

		// Serve the request using the Reverse Proxy
		proxy.ServeHTTP(w, r)
	})

	// Add simple health check endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status": "healthy", "service": "uco-proxy"}`))
	})

	// Configure HTTP server
	server := &http.Server{
		Addr:         ":8080",
		Handler:      mux,
		ReadTimeout:  120 * time.Second, // Long timeout to handle long prompt processing times
		WriteTimeout: 120 * time.Second, // Long timeout for streaming completions
	}

	log.Printf("[UCO Info] Universal Context Optimizer proxy listening on http://localhost:8080")
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Server failed to start: %v", err)
	}
}
