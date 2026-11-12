// Package reranker provides optional second-stage ranking for search results.
package reranker

import "context"

// Reranker scores documents against a query. Returned scores align with the
// input document order.
type Reranker interface {
	Rerank(ctx context.Context, query string, documents []string) ([]float32, error)
}
