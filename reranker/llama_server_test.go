package reranker

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/xavatu/grepai/config"
)

func TestLlamaServerRerank(t *testing.T) {
	var received rerankRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/rerank" {
			t.Errorf("path = %q, want /v1/rerank", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer secret" {
			t.Errorf("authorization = %q", got)
		}
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.2}]}`))
	}))
	defer server.Close()

	client := NewFromConfig(config.RerankerConfig{
		Enabled:               true,
		Endpoint:              server.URL + "/",
		Model:                 "test-reranker",
		APIKey:                "secret",
		RequestTimeoutSeconds: 1,
	})
	scores, err := client.Rerank(context.Background(), "find auth", []string{"doc-a", "doc-b"})
	if err != nil {
		t.Fatalf("Rerank() error = %v", err)
	}
	if !reflect.DeepEqual(scores, []float32{0.2, 0.9}) {
		t.Fatalf("scores = %v", scores)
	}
	if received.Model != "test-reranker" || received.Query != "find auth" || received.TopN != 2 {
		t.Fatalf("request metadata = %#v", received)
	}
	if !reflect.DeepEqual(received.Documents, []string{"doc-a", "doc-b"}) {
		t.Fatalf("documents = %v", received.Documents)
	}
}

func TestLlamaServerRerankRejectsInvalidResponses(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{name: "missing result", body: `{"results":[{"index":0,"relevance_score":0.2}]}`},
		{name: "duplicate index", body: `{"results":[{"index":0,"relevance_score":0.2},{"index":0,"relevance_score":0.9}]}`},
		{name: "out of range", body: `{"results":[{"index":0,"relevance_score":0.2},{"index":2,"relevance_score":0.9}]}`},
		{name: "malformed JSON", body: `{"results":`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = w.Write([]byte(tt.body))
			}))
			defer server.Close()

			client := NewFromConfig(config.RerankerConfig{Enabled: true, Endpoint: server.URL, RequestTimeoutSeconds: 1})
			if _, err := client.Rerank(context.Background(), "query", []string{"a", "b"}); err == nil {
				t.Fatal("Rerank() error = nil")
			}
		})
	}
}

func TestLlamaServerRerankHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "model unavailable", http.StatusServiceUnavailable)
	}))
	defer server.Close()

	client := NewFromConfig(config.RerankerConfig{Enabled: true, Endpoint: server.URL, RequestTimeoutSeconds: 1})
	if _, err := client.Rerank(context.Background(), "query", []string{"doc"}); err == nil {
		t.Fatal("Rerank() error = nil")
	}
}

func TestLlamaServerRerankHonorsCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(100 * time.Millisecond)
		_, _ = w.Write([]byte(`{"results":[{"index":0,"relevance_score":1}]}`))
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	client := NewFromConfig(config.RerankerConfig{Enabled: true, Endpoint: server.URL, RequestTimeoutSeconds: 1})
	_, err := client.Rerank(ctx, "query", []string{"doc"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Rerank() error = %v, want context.Canceled", err)
	}
}

func TestLlamaServerRerankEmptyDocuments(t *testing.T) {
	client := NewFromConfig(config.RerankerConfig{Enabled: true})
	scores, err := client.Rerank(context.Background(), "query", nil)
	if err != nil || len(scores) != 0 {
		t.Fatalf("Rerank() = %v, %v", scores, err)
	}
}
