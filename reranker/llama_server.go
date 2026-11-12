package reranker

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/yoanbernabeu/grepai/config"
)

const maxErrorBodyBytes = 4096

type llamaServerClient struct {
	endpoint string
	model    string
	apiKey   string
	client   *http.Client
}

type rerankRequest struct {
	Model     string   `json:"model,omitempty"`
	Query     string   `json:"query"`
	TopN      int      `json:"top_n"`
	Documents []string `json:"documents"`
}

type rerankResponse struct {
	Results []struct {
		Index          int     `json:"index"`
		RelevanceScore float32 `json:"relevance_score"`
	} `json:"results"`
}

// NewFromConfig creates a llama-server client when reranking is enabled.
func NewFromConfig(cfg config.RerankerConfig) Reranker {
	if !cfg.Enabled {
		return nil
	}
	endpoint := strings.TrimRight(cfg.Endpoint, "/")
	if endpoint == "" {
		endpoint = config.DefaultRerankerEndpoint
	}
	timeoutSeconds := cfg.RequestTimeoutSeconds
	if timeoutSeconds <= 0 {
		timeoutSeconds = config.DefaultRerankerTimeoutSeconds
	}
	return &llamaServerClient{
		endpoint: endpoint,
		model:    cfg.Model,
		apiKey:   cfg.APIKey,
		client: &http.Client{
			Timeout: time.Duration(timeoutSeconds) * time.Second,
		},
	}
}

func (c *llamaServerClient) Rerank(ctx context.Context, query string, documents []string) ([]float32, error) {
	if len(documents) == 0 {
		return []float32{}, nil
	}

	payload, err := json.Marshal(rerankRequest{
		Model:     c.model,
		Query:     query,
		TopN:      len(documents),
		Documents: documents,
	})
	if err != nil {
		return nil, fmt.Errorf("encode rerank request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint+"/v1/rerank", bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("create rerank request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, fmt.Errorf("call reranker: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodyBytes))
		return nil, fmt.Errorf("reranker returned HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var decoded rerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, fmt.Errorf("decode rerank response: %w", err)
	}
	if len(decoded.Results) != len(documents) {
		return nil, fmt.Errorf("reranker returned %d results for %d documents", len(decoded.Results), len(documents))
	}

	scores := make([]float32, len(documents))
	seen := make([]bool, len(documents))
	for _, result := range decoded.Results {
		if result.Index < 0 || result.Index >= len(documents) {
			return nil, fmt.Errorf("reranker returned out-of-range document index %d", result.Index)
		}
		if seen[result.Index] {
			return nil, fmt.Errorf("reranker returned duplicate document index %d", result.Index)
		}
		seen[result.Index] = true
		scores[result.Index] = result.RelevanceScore
	}

	return scores, nil
}
