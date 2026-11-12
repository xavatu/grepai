package mcp

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/qdrant/go-client/qdrant"
	"github.com/xavatu/grepai/config"
	"github.com/xavatu/grepai/store"
)

func TestServerQdrantStoreCacheIntegration(t *testing.T) {
	if os.Getenv("QDRANT_INTEGRATION") == "" {
		t.Skip("set QDRANT_INTEGRATION=1 to run against a local Qdrant instance")
	}

	port := 6334
	if raw := os.Getenv("QDRANT_PORT"); raw != "" {
		var err error
		port, err = strconv.Atoi(raw)
		if err != nil {
			t.Fatalf("invalid QDRANT_PORT: %v", err)
		}
	}
	endpoint := os.Getenv("QDRANT_ENDPOINT")
	if endpoint == "" {
		endpoint = "localhost"
	}
	host := strings.TrimPrefix(strings.TrimPrefix(endpoint, "http://"), "https://")
	if index := strings.IndexAny(host, ":/"); index >= 0 {
		host = host[:index]
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	collection := fmt.Sprintf("grepai_mcp_cache_%d", time.Now().UnixNano())
	admin, err := qdrant.NewClient(&qdrant.Config{Host: host, Port: port})
	if err != nil {
		t.Fatalf("create cleanup client: %v", err)
	}
	defer admin.Close()
	defer func() {
		if err := admin.DeleteCollection(context.Background(), collection); err != nil {
			t.Errorf("delete integration collection: %v", err)
		}
	}()

	denseOnDisk := false
	qdrantCfg := config.QdrantConfig{
		Endpoint:        endpoint,
		Port:            port,
		DenseOnDisk:     &denseOnDisk,
		UpsertBatchSize: 16,
	}
	dimensions := 4
	embedderCfg := config.EmbedderConfig{
		Provider:   "synthetic",
		Model:      "mcp-cache-integration",
		Dimensions: &dimensions,
	}

	server := &Server{}
	const callers = 8
	stores := make([]*store.QdrantStore, callers)
	errs := make([]error, callers)
	var wg sync.WaitGroup
	for i := range callers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			stores[i], errs[i] = server.qdrantStore(ctx, qdrantCfg, collection, embedderCfg, "")
		}()
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("caller %d failed: %v", i, err)
		}
		if stores[i] != stores[0] {
			t.Fatalf("caller %d received a different cached store", i)
		}
	}
	if len(server.qdrantStores) != 1 {
		t.Fatalf("cached stores = %d, want 1", len(server.qdrantStores))
	}

	first := stores[0]
	if err := server.Close(); err != nil {
		t.Fatalf("close cached stores: %v", err)
	}
	if len(server.qdrantStores) != 0 {
		t.Fatalf("cached stores after close = %d, want 0", len(server.qdrantStores))
	}
	if err := server.Close(); err != nil {
		t.Fatalf("second close should be idempotent: %v", err)
	}

	reopened, err := server.qdrantStore(ctx, qdrantCfg, collection, embedderCfg, "")
	if err != nil {
		t.Fatalf("reopen cached store: %v", err)
	}
	if reopened == first {
		t.Fatal("reopen returned the closed cached store")
	}
	if err := server.Close(); err != nil {
		t.Fatalf("close reopened store: %v", err)
	}
}
