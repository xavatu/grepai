package search

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/xavatu/grepai/config"
	"github.com/xavatu/grepai/store"
)

type searchTestEmbedder struct {
	vector []float32
}

func (e *searchTestEmbedder) Embed(context.Context, string) ([]float32, error) {
	return append([]float32(nil), e.vector...), nil
}

func (e *searchTestEmbedder) EmbedBatch(context.Context, []string) ([][]float32, error) {
	return nil, nil
}

func (e *searchTestEmbedder) Dimensions() int { return len(e.vector) }
func (e *searchTestEmbedder) Close() error    { return nil }

type searchTestStore struct {
	searchResults []store.SearchResult
	allChunks     []store.Chunk
	searchOptions store.SearchOptions
	searchCalls   int
	searchLimit   int
}

func (s *searchTestStore) SaveChunks(context.Context, []store.Chunk) error { return nil }
func (s *searchTestStore) DeleteByFile(context.Context, string) error      { return nil }
func (s *searchTestStore) Search(_ context.Context, _ []float32, limit int, opts store.SearchOptions) ([]store.SearchResult, error) {
	s.searchCalls++
	s.searchLimit = limit
	s.searchOptions = opts
	return append([]store.SearchResult(nil), s.searchResults...), nil
}

type searchTestReranker struct {
	scores    []float32
	err       error
	query     string
	documents []string
}

func (r *searchTestReranker) Rerank(_ context.Context, query string, documents []string) ([]float32, error) {
	r.query = query
	r.documents = append([]string(nil), documents...)
	return append([]float32(nil), r.scores...), r.err
}
func (s *searchTestStore) GetDocument(context.Context, string) (*store.Document, error) {
	return nil, nil
}
func (s *searchTestStore) SaveDocument(context.Context, store.Document) error { return nil }
func (s *searchTestStore) DeleteDocument(context.Context, string) error       { return nil }
func (s *searchTestStore) ListDocuments(context.Context) ([]string, error)    { return nil, nil }
func (s *searchTestStore) Load(context.Context) error                         { return nil }
func (s *searchTestStore) Persist(context.Context) error                      { return nil }
func (s *searchTestStore) Close() error                                       { return nil }
func (s *searchTestStore) GetStats(context.Context) (*store.IndexStats, error) {
	return &store.IndexStats{}, nil
}
func (s *searchTestStore) ListFilesWithStats(context.Context) ([]store.FileStats, error) {
	return nil, nil
}
func (s *searchTestStore) GetChunksForFile(context.Context, string) ([]store.Chunk, error) {
	return nil, nil
}
func (s *searchTestStore) GetAllChunks(context.Context) ([]store.Chunk, error) {
	return append([]store.Chunk(nil), s.allChunks...), nil
}

type nativeSearchTestStore struct {
	searchTestStore
	hybridCalls   int
	hybridQuery   string
	hybridVector  []float32
	hybridLimit   int
	hybridOptions store.SearchOptions
	hybridK       float32
}

func (s *nativeSearchTestStore) SearchHybrid(_ context.Context, query string, vector []float32, limit int, opts store.SearchOptions, k float32) ([]store.SearchResult, error) {
	s.hybridCalls++
	s.hybridQuery = query
	s.hybridVector = append([]float32(nil), vector...)
	s.hybridLimit = limit
	s.hybridOptions = opts
	s.hybridK = k
	return []store.SearchResult{{Chunk: store.Chunk{ID: "native"}, Score: 1}}, nil
}

func TestSearcherUsesNativeHybridStore(t *testing.T) {
	backend := &nativeSearchTestStore{}
	opts := store.SearchOptions{PathPrefixes: []string{"ws/api/", "ws/web/"}}
	searcher := NewSearcher(backend, &searchTestEmbedder{vector: []float32{1, 2}}, config.SearchConfig{
		Hybrid: config.HybridConfig{Enabled: true},
	})

	results, err := searcher.SearchWithOptions(context.Background(), "find token", 3, opts)
	if err != nil {
		t.Fatalf("SearchWithOptions failed: %v", err)
	}
	if backend.hybridCalls != 1 || backend.searchCalls != 0 {
		t.Fatalf("native calls = %d, dense calls = %d; want 1 and 0", backend.hybridCalls, backend.searchCalls)
	}
	if backend.hybridQuery != "find token" || !reflect.DeepEqual(backend.hybridVector, []float32{1, 2}) {
		t.Fatalf("native query = %q, vector = %v", backend.hybridQuery, backend.hybridVector)
	}
	if backend.hybridLimit != 6 {
		t.Fatalf("native limit = %d, want fetch limit 6", backend.hybridLimit)
	}
	if backend.hybridK != 60 || !reflect.DeepEqual(backend.hybridOptions, opts) {
		t.Fatalf("native k = %v, options = %#v", backend.hybridK, backend.hybridOptions)
	}
	if len(results) != 1 || results[0].Chunk.ID != "native" {
		t.Fatalf("results = %#v", results)
	}
}

func TestSearcherHybridFallbackHonorsMultiplePrefixes(t *testing.T) {
	backend := &searchTestStore{
		searchResults: []store.SearchResult{{Chunk: store.Chunk{ID: "dense", FilePath: "ws/api/dense.go"}, Score: 1}},
		allChunks: []store.Chunk{
			{ID: "api", FilePath: "ws/api/auth.go", Content: "validate token"},
			{ID: "web", FilePath: "ws/web/auth.ts", Content: "validate token"},
			{ID: "other", FilePath: "ws/other/auth.py", Content: "validate token"},
		},
	}
	opts := store.SearchOptions{PathPrefixes: []string{"ws/api/", "ws/web/"}}
	searcher := NewSearcher(backend, &searchTestEmbedder{vector: []float32{1, 2}}, config.SearchConfig{
		Hybrid: config.HybridConfig{Enabled: true, K: 60},
	})

	results, err := searcher.SearchWithOptions(context.Background(), "validate token", 10, opts)
	if err != nil {
		t.Fatalf("SearchWithOptions failed: %v", err)
	}
	if backend.searchCalls != 1 || !reflect.DeepEqual(backend.searchOptions, opts) {
		t.Fatalf("dense calls = %d, options = %#v", backend.searchCalls, backend.searchOptions)
	}
	for _, result := range results {
		if result.Chunk.ID == "other" {
			t.Fatalf("fallback returned chunk outside path filters: %#v", results)
		}
	}
}

func TestSearcherReranksCandidatesThenAppliesBoost(t *testing.T) {
	backend := &searchTestStore{searchResults: []store.SearchResult{
		{Chunk: store.Chunk{ID: "a1", FilePath: "src/a.go", StartLine: 1, EndLine: 3, Content: "alpha"}, Score: 0.9},
		{Chunk: store.Chunk{ID: "a2", FilePath: "src/a.go", StartLine: 8, EndLine: 9, Content: "duplicate"}, Score: 0.8},
		{Chunk: store.Chunk{ID: "b", FilePath: "pkg/tests/b.go", StartLine: 4, EndLine: 7, Content: "beta"}, Score: 0.7},
		{Chunk: store.Chunk{ID: "c", FilePath: "pkg/c.go", StartLine: 10, EndLine: 11, Content: "gamma"}, Score: 0.6},
		{Chunk: store.Chunk{ID: "d", FilePath: "pkg/d.go", Content: "delta"}, Score: 0.5},
	}}
	ranker := &searchTestReranker{scores: []float32{0.2, 0.9, 0.5}}
	searcher := NewSearcher(backend, &searchTestEmbedder{vector: []float32{1}}, config.SearchConfig{
		Dedup: config.DedupConfig{Enabled: true},
		Boost: config.BoostConfig{
			Enabled: true,
			Penalties: []config.BoostRule{
				{Pattern: "/tests/", Factor: 0.5},
			},
		},
	}, WithReranker(ranker, 3))

	results, err := searcher.Search(context.Background(), "find code", 2, "")
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if backend.searchLimit != 12 {
		t.Fatalf("store limit = %d, want 12", backend.searchLimit)
	}
	if len(ranker.documents) != 3 {
		t.Fatalf("reranked documents = %d, want 3", len(ranker.documents))
	}
	if ranker.query != "find code" || !strings.Contains(ranker.documents[0], "File: src/a.go\nLines: 1-3\nContent:\nalpha") {
		t.Fatalf("rerank input = query %q, documents %#v", ranker.query, ranker.documents)
	}
	if got := []string{results[0].Chunk.ID, results[1].Chunk.ID}; !reflect.DeepEqual(got, []string{"c", "b"}) {
		t.Fatalf("result order = %v, want [c b]", got)
	}
	if results[0].Score != 0.5 || results[1].Score != 0.45 {
		t.Fatalf("boosted scores = [%v %v]", results[0].Score, results[1].Score)
	}
}

func TestSearcherRerankerFailureFallsBackToRetrieval(t *testing.T) {
	backend := &searchTestStore{searchResults: []store.SearchResult{
		{Chunk: store.Chunk{ID: "a", FilePath: "src/a.go"}, Score: 0.9},
		{Chunk: store.Chunk{ID: "b", FilePath: "src/b.go"}, Score: 0.8},
	}}
	ranker := &searchTestReranker{err: errors.New("offline")}
	searcher := NewSearcher(backend, &searchTestEmbedder{vector: []float32{1}}, config.SearchConfig{}, WithReranker(ranker, 30))

	results, err := searcher.Search(context.Background(), "query", 1, "")
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) != 1 || results[0].Chunk.ID != "a" || results[0].Score != 0.9 {
		t.Fatalf("fallback results = %#v", results)
	}
}

func TestSearcherRerankerCancellationIsReturned(t *testing.T) {
	backend := &searchTestStore{searchResults: []store.SearchResult{
		{Chunk: store.Chunk{ID: "a", FilePath: "src/a.go"}, Score: 0.9},
	}}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	ranker := &searchTestReranker{err: context.Canceled}
	searcher := NewSearcher(backend, &searchTestEmbedder{vector: []float32{1}}, config.SearchConfig{}, WithReranker(ranker, 30))

	_, err := searcher.Search(ctx, "query", 1, "")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Search() error = %v, want context.Canceled", err)
	}
}
