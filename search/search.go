package search

import (
	"context"
	"fmt"
	"log"

	"github.com/yoanbernabeu/grepai/config"
	"github.com/yoanbernabeu/grepai/embedder"
	"github.com/yoanbernabeu/grepai/reranker"
	"github.com/yoanbernabeu/grepai/store"
)

type Searcher struct {
	store      store.VectorStore
	embedder   embedder.Embedder
	boostCfg   config.BoostConfig
	hybridCfg  config.HybridConfig
	dedupCfg   config.DedupConfig
	reranker   reranker.Reranker
	candidates int
}

// Option customizes a Searcher without breaking existing constructor callers.
type Option func(*Searcher)

// WithReranker enables second-stage ranking for up to candidates results.
func WithReranker(r reranker.Reranker, candidates int) Option {
	return func(s *Searcher) {
		s.reranker = r
		s.candidates = candidates
	}
}

func NewSearcher(st store.VectorStore, emb embedder.Embedder, searchCfg config.SearchConfig, options ...Option) *Searcher {
	s := &Searcher{
		store:     st,
		embedder:  emb,
		boostCfg:  searchCfg.Boost,
		hybridCfg: searchCfg.Hybrid,
		dedupCfg:  searchCfg.Dedup,
	}
	for _, option := range options {
		option(s)
	}
	return s
}

func (s *Searcher) Search(ctx context.Context, query string, limit int, pathPrefix string) ([]store.SearchResult, error) {
	return s.SearchWithOptions(ctx, query, limit, store.SearchOptions{PathPrefix: pathPrefix})
}

// SearchWithOptions searches with store-level filters. It is used by
// workspace search to push several project/path prefixes into the backend.
func (s *Searcher) SearchWithOptions(ctx context.Context, query string, limit int, opts store.SearchOptions) ([]store.SearchResult, error) {
	queryVector, err := s.embedder.Embed(ctx, query)
	if err != nil {
		return nil, err
	}

	if s.reranker != nil {
		return s.searchWithReranking(ctx, query, queryVector, limit, opts)
	}

	return s.searchWithoutReranking(ctx, query, queryVector, limit, opts)
}

func (s *Searcher) searchWithoutReranking(ctx context.Context, query string, queryVector []float32, limit int, opts store.SearchOptions) ([]store.SearchResult, error) {
	fetchMultiplier := 2
	if s.dedupCfg.Enabled {
		fetchMultiplier = 4
	}
	fetchLimit := limit * fetchMultiplier

	var (
		results []store.SearchResult
		err     error
	)

	if s.hybridCfg.Enabled {
		results, err = s.hybridSearch(ctx, query, queryVector, fetchLimit, opts)
	} else {
		results, err = s.store.Search(ctx, queryVector, fetchLimit, opts)
	}

	if err != nil {
		return nil, err
	}

	results = ApplyBoost(results, s.boostCfg)

	if s.dedupCfg.Enabled {
		results = DeduplicateByFile(results)
	}

	if len(results) > limit {
		results = results[:limit]
	}

	return results, nil
}

func (s *Searcher) searchWithReranking(ctx context.Context, query string, queryVector []float32, limit int, opts store.SearchOptions) ([]store.SearchResult, error) {
	candidateLimit := s.candidates
	if candidateLimit < limit {
		candidateLimit = limit
	}
	if candidateLimit < 1 {
		candidateLimit = config.DefaultRerankerCandidates
	}

	fetchLimit := candidateLimit
	if s.dedupCfg.Enabled {
		fetchLimit *= 4
	}

	var (
		results []store.SearchResult
		err     error
	)
	if s.hybridCfg.Enabled {
		results, err = s.hybridSearch(ctx, query, queryVector, fetchLimit, opts)
	} else {
		results, err = s.store.Search(ctx, queryVector, fetchLimit, opts)
	}
	if err != nil {
		return nil, err
	}

	if s.dedupCfg.Enabled {
		results = DeduplicateByFile(results)
	}
	if len(results) > candidateLimit {
		results = results[:candidateLimit]
	}
	if len(results) == 0 {
		return results, nil
	}

	documents := make([]string, len(results))
	for i, result := range results {
		documents[i] = formatRerankDocument(result)
	}

	scores, rerankErr := s.reranker.Rerank(ctx, query, documents)
	if rerankErr != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		log.Printf("Warning: reranker unavailable, using retrieval ranking: %v", rerankErr)
	} else if len(scores) != len(results) {
		log.Printf("Warning: reranker returned %d scores for %d results, using retrieval ranking", len(scores), len(results))
	} else {
		for i := range results {
			results[i].Score = scores[i]
		}
		store.SortResultsByScore(results)
	}

	results = ApplyBoost(results, s.boostCfg)
	if len(results) > limit {
		results = results[:limit]
	}
	return results, nil
}

func formatRerankDocument(result store.SearchResult) string {
	return fmt.Sprintf("File: %s\nLines: %d-%d\nContent:\n%s", result.Chunk.FilePath, result.Chunk.StartLine, result.Chunk.EndLine, result.Chunk.Content)
}

// hybridSearch combines vector search and text search using RRF.
func (s *Searcher) hybridSearch(ctx context.Context, query string, queryVector []float32, limit int, opts store.SearchOptions) ([]store.SearchResult, error) {
	if native, ok := s.store.(store.NativeHybridStore); ok {
		k := s.hybridCfg.K
		if k <= 0 {
			k = 60
		}
		return native.SearchHybrid(ctx, query, queryVector, limit, opts, k)
	}

	vectorResults, err := s.store.Search(ctx, queryVector, limit, opts)
	if err != nil {
		return nil, err
	}

	allChunks, err := s.store.GetAllChunks(ctx)
	if err != nil {
		return nil, err
	}

	textResults := TextSearchWithOptions(ctx, allChunks, query, limit, opts)

	k := s.hybridCfg.K
	if k <= 0 {
		k = 60
	}

	return ReciprocalRankFusion(k, limit, vectorResults, textResults), nil
}
