package store

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestQdrantStoreIntegration(t *testing.T) {
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

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	collection := fmt.Sprintf("grepai_integration_%d", time.Now().UnixNano())
	st, err := NewQdrantStore(ctx, endpoint, port, false, collection, "", 4,
		WithQdrantDenseOnDisk(false),
		WithQdrantUpsertBatchSize(64),
		WithQdrantEmbeddingCacheNamespace("integration-model:4"))
	if err != nil {
		t.Fatalf("create store: %v", err)
	}
	defer st.Close()
	defer func() {
		if err := st.client.DeleteCollection(context.Background(), collection); err != nil {
			t.Errorf("delete integration collection: %v", err)
		}
	}()

	now := time.Now().UTC().Truncate(time.Second)
	chunks := make([]Chunk, 0, defaultScrollPage+3)
	for i := 0; i < defaultScrollPage+3; i++ {
		path := "workspace/api/src/generated.go"
		content := "ordinary generated fixture"
		vector := []float32{0, 1, 0, 0}
		if i == 0 {
			path = "workspace/api/src/auth.go"
			content = "func validateQuasarToken() error"
			vector = []float32{1, 0, 0, 0}
		} else if i == 1 {
			path = "workspace/web/src/auth.ts"
			content = "export function loginUser()"
		}
		chunks = append(chunks, Chunk{
			ID:          fmt.Sprintf("chunk-%03d", i),
			FilePath:    path,
			StartLine:   i + 1,
			EndLine:     i + 1,
			Content:     content,
			Vector:      vector,
			Hash:        fmt.Sprintf("hash-%03d", i),
			ContentHash: fmt.Sprintf("content-%03d", i),
			UpdatedAt:   now,
		})
	}
	if err := st.SaveChunks(ctx, chunks); err != nil {
		t.Fatalf("save chunks: %v", err)
	}
	for _, doc := range []Document{
		{Path: "workspace/api/src/auth.go", Hash: "doc-api", ModTime: now, ChunkIDs: []string{"chunk-000"}},
		{Path: "workspace/web/src/auth.ts", Hash: "doc-web", ModTime: now, ChunkIDs: []string{"chunk-001"}},
	} {
		if err := st.SaveDocument(ctx, doc); err != nil {
			t.Fatalf("save document: %v", err)
		}
	}

	all, err := st.GetAllChunks(ctx)
	if err != nil {
		t.Fatalf("get paginated chunks: %v", err)
	}
	if got, want := len(all), defaultScrollPage+3; got != want {
		t.Fatalf("paginated chunk count = %d, want %d", got, want)
	}

	results, err := st.SearchHybrid(ctx, "validateQuasarToken", []float32{0, 1, 0, 0}, 10,
		SearchOptions{PathPrefixes: []string{"workspace/api/src/", "workspace/web/src/"}}, 60)
	if err != nil {
		t.Fatalf("hybrid search: %v", err)
	}
	identifierRank := -1
	for i, result := range results {
		if result.Chunk.ID == "chunk-000" {
			identifierRank = i
			break
		}
	}
	// The best dense-only result and best BM25-only result have the same RRF
	// score, so Qdrant may return either first. The lexical hit must still beat
	// the remaining dense candidates even though hundreds share their vector.
	if identifierRank < 0 || identifierRank > 1 {
		t.Fatalf("BM25 identifier rank = %d, want 0 or 1: %#v", identifierRank, results)
	}

	filtered, err := st.Search(ctx, []float32{0, 1, 0, 0}, 10, SearchOptions{PathPrefixes: []string{"workspace/web/"}})
	if err != nil {
		t.Fatalf("filtered search: %v", err)
	}
	if len(filtered) != 1 || filtered[0].Chunk.ID != "chunk-001" {
		t.Fatalf("server-side path filter returned unexpected results: %#v", filtered)
	}

	doc, err := st.GetDocument(ctx, "workspace/api/src/auth.go")
	if err != nil || doc == nil || doc.Hash != "doc-api" || len(doc.ChunkIDs) != 1 {
		t.Fatalf("document round trip failed: doc=%#v err=%v", doc, err)
	}
	stats, err := st.GetStats(ctx)
	if err != nil || stats.TotalFiles != 2 || stats.TotalChunks != defaultScrollPage+3 {
		t.Fatalf("stats mismatch: stats=%#v err=%v", stats, err)
	}
	vector, found, err := st.LookupByContentHash(ctx, "content-000", "")
	if err != nil || !found || len(vector) != 4 {
		t.Fatalf("embedding cache lookup failed: found=%v vector=%v err=%v", found, vector, err)
	}

	documents, err := st.ListDocuments(ctx)
	if err != nil || len(documents) != 2 || documents[0] != "workspace/api/src/auth.go" || documents[1] != "workspace/web/src/auth.ts" {
		t.Fatalf("document listing mismatch: documents=%v err=%v", documents, err)
	}
	fileChunks, err := st.GetChunksForFile(ctx, "workspace/api/src/auth.go")
	if err != nil || len(fileChunks) != 1 || fileChunks[0].ID != "chunk-000" || len(fileChunks[0].Vector) != 4 {
		t.Fatalf("file chunk round trip failed: chunks=%#v err=%v", fileChunks, err)
	}
	fileStats, err := st.ListFilesWithStats(ctx)
	if err != nil || len(fileStats) != 2 || fileStats[0].Path != "workspace/api/src/auth.go" || fileStats[0].ChunkCount != 1 {
		t.Fatalf("file stats mismatch: stats=%#v err=%v", fileStats, err)
	}

	denseFallback, err := st.SearchHybrid(ctx, " ", []float32{1, 0, 0, 0}, 1,
		SearchOptions{PathPrefixes: []string{"workspace/api/src/auth.go"}}, 60)
	if err != nil || len(denseFallback) != 1 || denseFallback[0].Chunk.ID != "chunk-000" {
		t.Fatalf("empty-query dense fallback mismatch: results=%#v err=%v", denseFallback, err)
	}

	_, err = NewQdrantStore(ctx, endpoint, port, false, collection, "", 4,
		WithQdrantEmbeddingCacheNamespace("different-model:4"))
	if err == nil || !strings.Contains(err.Error(), "embedding model/cache namespace mismatch") {
		t.Fatalf("incompatible cache namespace error = %v", err)
	}

	if err := st.DeleteByFile(ctx, "workspace/api/src/auth.go"); err != nil {
		t.Fatalf("delete chunks: %v", err)
	}
	if err := st.DeleteDocument(ctx, "workspace/api/src/auth.go"); err != nil {
		t.Fatalf("delete document: %v", err)
	}
	deletedChunks, err := st.GetChunksForFile(ctx, "workspace/api/src/auth.go")
	if err != nil || len(deletedChunks) != 0 {
		t.Fatalf("deleted chunks remain: chunks=%#v err=%v", deletedChunks, err)
	}
	deletedDoc, err := st.GetDocument(ctx, "workspace/api/src/auth.go")
	if err != nil || deletedDoc != nil {
		t.Fatalf("deleted document remains: doc=%#v err=%v", deletedDoc, err)
	}
	stats, err = st.GetStats(ctx)
	if err != nil || stats.TotalFiles != 1 || stats.TotalChunks != defaultScrollPage+2 {
		t.Fatalf("stats after deletion mismatch: stats=%#v err=%v", stats, err)
	}
}
