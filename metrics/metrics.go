package metrics

import (
	"encoding/json"
	"log"
	"strings"

	"uco-proxy/router"
)

// RequestStats represents the complete cost and token delta breakdown for a single API request.
type RequestStats struct {
	Model                 string  `json:"model"`
	OriginalTextTokens    int     `json:"original_text_tokens"`
	OptimizedVisionTokens int     `json:"optimized_vision_tokens"`
	InputRatePerM         float64 `json:"input_rate_per_m_usd"`
	OriginalCostUSD       float64 `json:"original_cost_usd"`
	OptimizedCostUSD      float64 `json:"optimized_cost_usd"`
	CostSavingsUSD        float64 `json:"cost_savings_usd"`
	SavingsPercentage     float64 `json:"savings_percentage"`
}

// GetInputRate returns the input token cost per million tokens based on model provider pricing.
func GetInputRate(model string) float64 {
	m := strings.ToLower(model)
	if strings.Contains(m, "claude") {
		return 3.00 // Anthropic Claude 3.5 Sonnet input cost is $3.00 / M tokens
	}
	// Default to OpenAI GPT-4o input cost: $2.50 / M tokens
	return 2.50
}

// ComputeStats calculates input token totals and cost savings for the request.
func ComputeStats(model string, segments []router.MessageSegment) RequestStats {
	originalTextTokens := 0
	optimizedVisionTokens := 0

	for _, seg := range segments {
		originalTextTokens += seg.TextTokens
		if seg.Strategy == router.RenderBitmap {
			optimizedVisionTokens += seg.EstimatedVisionTokens
		} else {
			optimizedVisionTokens += seg.TextTokens
		}
	}

	rate := GetInputRate(model)
	originalCost := (float64(originalTextTokens) / 1_000_000.0) * rate
	optimizedCost := (float64(optimizedVisionTokens) / 1_000_000.0) * rate
	savings := originalCost - optimizedCost

	pct := 0.0
	if originalCost > 0 {
		pct = (savings / originalCost) * 100.0
	}

	return RequestStats{
		Model:                 model,
		OriginalTextTokens:    originalTextTokens,
		OptimizedVisionTokens: optimizedVisionTokens,
		InputRatePerM:         rate,
		OriginalCostUSD:       originalCost,
		OptimizedCostUSD:      optimizedCost,
		CostSavingsUSD:        savings,
		SavingsPercentage:     pct,
	}
}

// LogTelemetry outputs the computed stats as structured JSON telemetry to standard logger.
func LogTelemetry(stats RequestStats) {
	data, err := json.Marshal(stats)
	if err == nil {
		log.Printf("[UCO Telemetry] %s", string(data))
	} else {
		log.Printf("[UCO Error] Failed to marshal telemetry: %v", err)
	}
}
