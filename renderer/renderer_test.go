package renderer

import (
	"context"
	"crypto/sha256"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"uco-proxy/cache"
)

// TestRenderTextToPNG checks that rendering succeeds and outputs valid PNG bytes.
func TestRenderTextToPNG(t *testing.T) {
	sampleText := `#!/usr/bin/env python
print("Hello, world!")
for i in range(10):
    print(i)
`
	pngBytes, err := RenderTextToPNG(sampleText)
	if err != nil {
		t.Fatalf("Failed to render text to PNG: %v", err)
	}

	if len(pngBytes) < 100 {
		t.Errorf("Expected non-empty PNG buffer, got %d bytes", len(pngBytes))
	}
}

// BenchmarkRenderTextToPNG benchmarks the image generation latency.
// Milestone 4 requires this to take under 15ms.
func BenchmarkRenderTextToPNG(b *testing.B) {
	sampleText := `package main

import "fmt"

func main() {
	fmt.Println("This is a benchmark file representing a medium sized code file.")
	fmt.Println("It should render in Go under 15 milliseconds.")
	for i := 0; i < 50; i++ {
		fmt.Printf("Line index: %d\n", i)
	}
}
`
	// Warm up once to load the font into memory
	_, _ = RenderTextToPNG(sampleText)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := RenderTextToPNG(sampleText)
		if err != nil {
			b.Fatalf("Render failed in benchmark: %v", err)
		}
	}
}

// TestConcurrentCache validates the cache orchestration under heavy concurrent load.
func TestConcurrentCache(t *testing.T) {
	// Initialize local in-memory cache
	localCache := cache.NewInMemoryCache()
	ctx := context.Background()

	sampleText := "func test() { return 42 }"
	hashBytes := sha256.Sum256([]byte(sampleText))
	hashKey := fmt.Sprintf("%x", hashBytes)

	var cacheHits int64
	var cacheMisses int64
	var renderCalls int64

	numGoroutines := 20
	var wg sync.WaitGroup

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			// Simulate small delay variation
			time.Sleep(time.Duration(id%3) * time.Millisecond)

			// Step 1: Check cache
			data, err := localCache.Get(ctx, hashKey)
			if err != nil {
				t.Errorf("Cache lookup failed: %v", err)
				return
			}

			if data != nil {
				// Cache HIT
				atomic.AddInt64(&cacheHits, 1)
			} else {
				// Cache MISS
				atomic.AddInt64(&cacheMisses, 1)

				// Simulate rendering step (render once and write to cache)
				atomic.AddInt64(&renderCalls, 1)
				pngBytes, err := RenderTextToPNG(sampleText)
				if err != nil {
					t.Errorf("Render failed: %v", err)
					return
				}

				// Write back to cache
				err = localCache.Set(ctx, hashKey, pngBytes, 1*time.Hour)
				if err != nil {
					t.Errorf("Cache write failed: %v", err)
				}
			}
		}(i)
	}

	wg.Wait()

	t.Logf("Concurrency results - Total requests: %d, Hits: %d, Misses: %d, Renders: %d",
		numGoroutines, cacheHits, cacheMisses, renderCalls)

	// In a concurrent cache model, we expect at least one miss (the first request)
	// and subsequent requests should hit the cache.
	if cacheMisses == 0 {
		t.Error("Expected at least 1 cache miss (cold start), got 0")
	}
	if cacheHits == 0 {
		t.Error("Expected concurrent requests to hit the cache, got 0 hits")
	}
}
