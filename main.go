package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"time"

	"uco-proxy/adapters"
	"uco-proxy/cache"
	"uco-proxy/db"
	"uco-proxy/metrics"
	"uco-proxy/renderer"
	"uco-proxy/router"
)

func main() {
	dbPath := os.Getenv("DB_PATH")
	if dbPath == "" {
		dbPath = "uco.db"
	}
	db.InitDB(dbPath)

	// Initialize cache with fallback (attempts localhost:6379)
	cache.InitCache("localhost:6379")

	upstream := os.Getenv("UPSTREAM_URL")
	if upstream == "" {
		upstream = "https://api.openai.com"
	}
	targetURL, err := url.Parse(upstream)
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

	// Authentication Middleware
	authMiddleware := func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			var key string
			isAuthBearer := false

			// Check custom X-UCO-API-Key header first
			if ucoKey := r.Header.Get("X-UCO-API-Key"); ucoKey != "" {
				key = ucoKey
			} else if authHeader := r.Header.Get("Authorization"); strings.HasPrefix(authHeader, "Bearer ") {
				key = strings.TrimPrefix(authHeader, "Bearer ")
				isAuthBearer = true
			}

			if key == "" {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusUnauthorized)
				_, _ = w.Write([]byte(`{"error": {"message": "Missing API Key. Provide X-UCO-API-Key or Bearer Token."}}`))
				return
			}

			// Validate against database
			valid, clientName, err := db.ValidateKey(key)
			if err != nil {
				log.Printf("[UCO Error] DB error validating key: %v", err)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = w.Write([]byte(`{"error": {"message": "Internal Database Error."}}`))
				return
			}

			if !valid {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusUnauthorized)
				_, _ = w.Write([]byte(`{"error": {"message": "Invalid or inactive UCO API Key."}}`))
				return
			}

			log.Printf("[UCO Auth] Client '%s' successfully authenticated.", clientName)

			// If the user set UCO API key in Authorization header, we must override it
			// with the real server-side upstream key so that the call to OpenAI succeeds.
			if isAuthBearer {
				upstreamKey := os.Getenv("OPENAI_API_KEY")
				if upstreamKey != "" {
					r.Header.Set("Authorization", "Bearer "+upstreamKey)
				}
			}

			next(w, r)
		}
	}

	// Setup Server Multiplexer
	mux := http.NewServeMux()

	// Intercept OpenAI chat completions route, protected by Auth Middleware
	mux.HandleFunc("/v1/chat/completions", authMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusMethodNotAllowed)
			_, _ = w.Write([]byte(`{"error": {"message": "Only POST requests are supported on /v1/chat/completions"}}`))
			return
		}

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
			proxy.ServeHTTP(w, r)
			return
		}

		// Run routing analysis using token budgeting
		analysis := router.AnalyzePayload(payload, router.DefaultTokenCounter)
		log.Printf("[UCO Info] ContextRouter Analysis for model '%s':", analysis.Model)

		// Compute request cost savings & log structured telemetry JSON
		stats := metrics.ComputeStats(payload.Model, analysis.Segments)
		metrics.LogTelemetry(stats)

		// Persist request execution metrics into SQLite Database
		err = db.LogRequestMetric(stats.Model, stats.OriginalTextTokens, stats.OptimizedVisionTokens, stats.CostSavingsUSD)
		if err != nil {
			log.Printf("[UCO Error] Failed to write request metrics to database: %v", err)
		}

		imageBuffers := make(map[int][]byte)
		hasOptimization := false

		for idx, seg := range analysis.Segments {
			log.Printf("  - Msg %d [%s] (Static: %t) -> Strategy: %s (Text: %d T, Vision Est: %d T)",
				idx, seg.Role, seg.IsStatic, seg.Strategy, seg.TextTokens, seg.EstimatedVisionTokens)

			// Orchestrate Milestone 4 Caching Flow if Strategy is RENDER_BITMAP
			if seg.Strategy == router.RenderBitmap {
				// Step 1: Compute SHA-256 hash of static text segment
				hash := sha256.Sum256([]byte(seg.ContentText))
				hashKey := fmt.Sprintf("uco:img:%x", hash)

				// Step 2: Query the Cache
				cachedBytes, err := cache.GlobalCache.Get(r.Context(), hashKey)
				if err != nil {
					log.Printf("[UCO Error] Cache lookup failed for Msg %d: %v", idx, err)
				}

				if cachedBytes != nil {
					// Cache HIT
					log.Printf("[UCO Info] Cache HIT for Msg %d (hash: %s). Bypassing renderer.", idx, hashKey[:16])
					imageBuffers[idx] = cachedBytes
					hasOptimization = true
				} else {
					// Cache MISS
					log.Printf("[UCO Info] Cache MISS for Msg %d (hash: %s). Rendering text to image...", idx, hashKey[:16])

					renderStart := time.Now()
					pngBytes, err := renderer.RenderTextToPNG(seg.ContentText)
					renderDuration := time.Since(renderStart)

					if err != nil {
						log.Printf("[UCO Error] Rendering failed for Msg %d: %v", idx, err)
					} else {
						log.Printf("[UCO Info] Rendered Msg %d in %v (size: %d bytes)", idx, renderDuration, len(pngBytes))
						imageBuffers[idx] = pngBytes
						hasOptimization = true

						// Step 3: Write to cache asynchronously with 24h expiration
						go func(key string, val []byte) {
							bgCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
							defer cancel()
							if err := cache.GlobalCache.Set(bgCtx, key, val, 24*time.Hour); err != nil {
								log.Printf("[UCO Error] Async cache write failed for %s: %v", key, err)
							} else {
								log.Printf("[UCO Info] Async cache write successful for %s", key[:16])
							}
						}(hashKey, pngBytes)
					}
				}
			}
		}

		// Mutate the payload if optimization occurred
		if hasOptimization {
			adapter := adapters.GetAdapter(payload.Model)
			mutatedBody, err := adapter.AdaptPayload(r.Context(), analysis.Segments, imageBuffers, bodyBytes)
			if err != nil {
				log.Printf("[UCO Error] Failed to adapt payload: %v. Forwarding original payload.", err)
				r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
			} else {
				log.Printf("[UCO Info] Mutated request payload size: %d -> %d bytes", len(bodyBytes), len(mutatedBody))
				// Replace request body with mutated payload
				r.Body = io.NopCloser(bytes.NewBuffer(mutatedBody))
				r.ContentLength = int64(len(mutatedBody))
				r.Header.Set("Content-Length", fmt.Sprintf("%d", len(mutatedBody)))
			}
		} else {
			// No optimization: restore original request body
			r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
		}

		// Serve the request using the Reverse Proxy
		proxy.ServeHTTP(w, r)
	})))

	// Production health check monitoring DB, Cache, and Upstream connectivity status
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		dbStatus := "online"
		if err := db.Ping(); err != nil {
			dbStatus = "offline: " + err.Error()
		}

		cacheStatus := "online"
		cacheProvider := "Redis"
		ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
		defer cancel()
		_, cacheErr := cache.GlobalCache.Get(ctx, "uco:healthcheck")
		if cacheErr != nil {
			cacheStatus = "offline: " + cacheErr.Error()
		}
		if _, ok := cache.GlobalCache.(*cache.InMemoryCache); ok {
			cacheProvider = "InMemory"
		}

		// Check upstream connection (use cheap GET request to verify)
		upstreamStatus := "online"
		client := &http.Client{Timeout: 1 * time.Second}
		resp, err := client.Get("https://api.openai.com/v1/models")
		if err != nil {
			upstreamStatus = "offline: " + err.Error()
		} else {
			resp.Body.Close()
		}

		status := http.StatusOK
		if dbStatus != "online" || upstreamStatus != "online" {
			status = http.StatusServiceUnavailable
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)

		response := map[string]interface{}{
			"status":   "healthy",
			"database": dbStatus,
			"upstream": upstreamStatus,
			"cache": map[string]string{
				"provider": cacheProvider,
				"status":   cacheStatus,
			},
			"timestamp": time.Now().Format(time.RFC3339),
		}

		if status != http.StatusOK {
			response["status"] = "unhealthy"
		}

		jsonData, _ := json.Marshal(response)
		_, _ = w.Write(jsonData)
	})

	// Configure HTTP server
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	server := &http.Server{
		Addr:         ":" + port,
		Handler:      mux,
		ReadTimeout:  120 * time.Second, // Long timeout to handle long prompt processing times
		WriteTimeout: 120 * time.Second, // Long timeout for streaming completions
	}

	log.Printf("[UCO Info] Universal Context Optimizer proxy listening on http://localhost:%s", port)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Server failed to start: %v", err)
	}
}
