package store

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/google/uuid"
	"github.com/qdrant/go-client/qdrant"
	"github.com/yoanbernabeu/grepai/git"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	qdrantSchemaVersion = 2
	denseVectorName     = "dense"
	bm25VectorName      = "bm25"
	metadataVectorName  = "metadata"
	bm25ModelName       = "qdrant/bm25"
	defaultUpsertBatch  = 128
	defaultScrollPage   = 512

	// qdrantLookupLimit bounds the points (and vectors) read by one
	// LookupByContentHash call.
	qdrantLookupLimit = 64

	payloadRecordType    = "record_type"
	payloadChunkID       = "chunk_id"
	payloadFilePath      = "file_path"
	payloadStartLine     = "start_line"
	payloadEndLine       = "end_line"
	payloadContent       = "content"
	payloadHash          = "hash"
	payloadFileHash      = "file_hash"
	payloadContentHash   = "content_hash"
	payloadCacheNS       = "embedding_cache_namespace"
	payloadUpdatedAt     = "updated_at"
	payloadDocumentHash  = "document_hash"
	payloadDocumentMTime = "document_mod_time"
	payloadChunkIDs      = "chunk_ids"

	// qdrantNamespaceField is the payload field holding a point's project
	// namespace in a collection that several projects may share (#247).
	qdrantNamespaceField = "project_namespace"
	// metadataProjectNamespaces marks collections whose point IDs and payloads
	// carry the project namespace. Schema v2 collections created before it keep
	// their layout: their points have no namespace and are read as before.
	metadataProjectNamespaces = "project_namespaces"

	recordChunk    = "chunk"
	recordDocument = "document"
)

var qdrantUUIDNamespace = uuid.MustParse("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

// qdrantNamespaces caches QdrantProjectNamespace per project root, so a
// long-running process (MCP server) does not run git on every store it opens.
var qdrantNamespaces sync.Map

// QdrantProjectNamespace returns the namespace a project uses inside its
// Qdrant collection. A collection derived from the project path belongs to that
// project alone and needs none. An explicitly configured collection can be
// shared by several projects, so their points are kept apart (#247) by
// repository: the normalized origin remote URL (plus the project's directory
// inside the repository), so that clones on other machines and linked
// worktrees of one repository still share an index. Without a git repository
// or a remote that names one, the resolved project root is used; when git is
// missing or fails (timeout, untrusted repository) too, with a warning, since
// processes that can read the repository then use another namespace.
func QdrantProjectNamespace(configuredCollection, projectRoot string) string {
	if configuredCollection == "" {
		return ""
	}
	if resolved, err := filepath.EvalSymlinks(projectRoot); err == nil {
		projectRoot = resolved
	}
	if abs, err := filepath.Abs(projectRoot); err == nil {
		projectRoot = abs
	}
	projectRoot = filepath.Clean(projectRoot)
	if cached, ok := qdrantNamespaces.Load(projectRoot); ok {
		if namespace, ok := cached.(string); ok {
			return namespace
		}
	}
	namespace := projectRoot
	identity, ok, err := git.RepoIdentity(projectRoot)
	switch {
	case errors.Is(err, exec.ErrNotFound):
		log.Printf("Warning: git is not installed: the shared Qdrant collection %q keeps %s apart by its path, not by its repository", configuredCollection, projectRoot)
	case err != nil:
		// Refusing to start would break setups that worked before (e.g. a
		// container running as another user than the repository owner).
		log.Printf("Warning: cannot identify the git repository of %s (%v): the shared Qdrant collection %q keeps it apart by its path, so grepai processes that can read the repository use another namespace", projectRoot, err, configuredCollection)
		return namespace // not cached: the next store retries git
	case ok:
		namespace = identity
	}
	qdrantNamespaces.Store(projectRoot, namespace)
	return namespace
}

// QdrantStoreOption customizes the Qdrant collection and write path.
type QdrantStoreOption func(*QdrantStore)

// WithQdrantDenseOnDisk serves original dense vectors from disk in collections
// this store creates.
func WithQdrantDenseOnDisk(enabled bool) QdrantStoreOption {
	return func(s *QdrantStore) { s.denseOnDisk = enabled }
}

// WithQdrantUpsertBatchSize sets the number of points per synchronous upsert.
func WithQdrantUpsertBatchSize(size int) QdrantStoreOption {
	return func(s *QdrantStore) {
		if size > 0 {
			s.upsertBatchSize = size
		}
	}
}

// WithQdrantEmbeddingCacheNamespace ties the collection to one embedding
// model, so vectors of another model are never reused or mixed in.
func WithQdrantEmbeddingCacheNamespace(namespace string) QdrantStoreOption {
	return func(s *QdrantStore) {
		if strings.TrimSpace(namespace) != "" {
			s.embeddingCacheNamespace = namespace
		}
	}
}

// WithQdrantProjectNamespace limits reads and writes to one project inside a
// collection shared by several projects (see QdrantProjectNamespace). It has
// no effect on collections created before project namespaces existed.
func WithQdrantProjectNamespace(namespace string) QdrantStoreOption {
	return func(s *QdrantStore) { s.namespace = namespace }
}

type QdrantStore struct {
	client                  *qdrant.Client
	collectionName          string
	dimensions              int
	apiKey                  string
	address                 string // host:port, for error messages
	denseOnDisk             bool
	upsertBatchSize         int
	embeddingCacheNamespace string
	// namespace scopes every point to one project. Empty keeps point IDs and
	// payloads of a collection that belongs to a single project.
	namespace string
}

func parseHost(endpoint string) string {
	host := strings.TrimPrefix(endpoint, "http://")
	host = strings.TrimPrefix(host, "https://")
	if idx := strings.Index(host, ":"); idx != -1 {
		host = host[:idx]
	}
	if idx := strings.Index(host, "/"); idx != -1 {
		host = host[:idx]
	}
	return host
}

func NewQdrantStore(ctx context.Context, endpoint string, port int, useTLS bool, collection, apiKey string, dimensions int, opts ...QdrantStoreOption) (*QdrantStore, error) {
	if dimensions <= 0 {
		return nil, fmt.Errorf("dimensions must be positive, got: %d", dimensions)
	}
	if port <= 0 {
		port = 6334
	}
	host := parseHost(endpoint)

	client, err := qdrant.NewClient(&qdrant.Config{
		Host:   host,
		Port:   port,
		UseTLS: useTLS,
		APIKey: apiKey,
		GrpcOptions: []grpc.DialOption{
			grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(64 * 1024 * 1024)),
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create Qdrant client: %w", err)
	}

	st := &QdrantStore{
		client:                  client,
		collectionName:          collection,
		dimensions:              dimensions,
		apiKey:                  apiKey,
		address:                 fmt.Sprintf("%s:%d", host, port),
		denseOnDisk:             true,
		upsertBatchSize:         defaultUpsertBatch,
		embeddingCacheNamespace: fmt.Sprintf("embedding-cache-v2:dimensions=%d", dimensions),
	}
	for _, opt := range opts {
		opt(st)
	}

	if err := st.ensureCollection(ctx); err != nil {
		_ = client.Close()
		return nil, err
	}
	return st, nil
}

// connectionError explains the most common cause of a failed first call:
// Qdrant not running, or grepai pointed at the REST port instead of gRPC (#177).
func (s *QdrantStore) connectionError(err error) error {
	hint := ""
	switch {
	case strings.Contains(err.Error(), "HTTP/1.1"):
		hint = "\nThe server answered with HTTP/1.1: grepai talks to Qdrant over gRPC, so use the gRPC port (6334 by default), not the REST port (6333)."
	case status.Code(err) == codes.Unavailable || status.Code(err) == codes.DeadlineExceeded:
		hint = "\nIs Qdrant running? Start it with: docker run -p 6334:6334 qdrant/qdrant\nOtherwise check store.qdrant.endpoint and store.qdrant.port in .grepai/config.yaml."
	}
	return fmt.Errorf("cannot reach Qdrant at %s: %w%s", s.address, err, hint)
}

func (s *QdrantStore) ensureCollection(ctx context.Context) error {
	exists, err := s.client.CollectionExists(ctx, s.collectionName)
	if err != nil {
		return s.connectionError(err)
	}
	if !exists {
		if err := s.createCollection(ctx); err != nil {
			return err
		}
	}

	info, err := s.client.GetCollectionInfo(ctx, s.collectionName)
	if err != nil {
		return fmt.Errorf("failed to inspect Qdrant collection: %w", err)
	}
	if err := s.validateCollection(info); err != nil {
		return err
	}
	if !info.Config.Metadata[metadataProjectNamespaces].GetBoolValue() {
		// Points of an older v2 collection carry no namespace: filtering on
		// one would hide them all. Rebuild the collection to share it.
		s.namespace = ""
	}
	return s.ensurePayloadIndexes(ctx, info)
}

func (s *QdrantStore) createCollection(ctx context.Context) error {
	metadata := qdrant.NewValueMap(map[string]any{
		"grepai_schema_version":     qdrantSchemaVersion,
		"embedding_dimensions":      s.dimensions,
		"embedding_cache_namespace": s.embeddingCacheNamespace,
		metadataProjectNamespaces:   true,
	})
	err := s.client.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: s.collectionName,
		VectorsConfig: qdrant.NewVectorsConfigMap(map[string]*qdrant.VectorParams{
			denseVectorName: {
				Size:     uint64(s.dimensions),
				Distance: qdrant.Distance_Cosine,
				OnDisk:   qdrant.PtrOf(s.denseOnDisk),
			},
		}),
		SparseVectorsConfig: qdrant.NewSparseVectorsConfig(map[string]*qdrant.SparseVectorParams{
			bm25VectorName:     {Modifier: qdrant.Modifier_Idf.Enum()},
			metadataVectorName: {},
		}),
		ShardNumber:       qdrant.PtrOf(uint32(1)),
		ReplicationFactor: qdrant.PtrOf(uint32(1)),
		Metadata:          metadata,
	})
	if err != nil && status.Code(err) != codes.AlreadyExists {
		return fmt.Errorf("failed to create Qdrant collection: %w", err)
	}
	return nil
}

func (s *QdrantStore) validateCollection(info *qdrant.CollectionInfo) error {
	if info == nil || info.Config == nil || info.Config.Params == nil {
		return s.schemaError("collection configuration is missing")
	}
	metadata := info.Config.Metadata
	if metadata["grepai_schema_version"].GetIntegerValue() != qdrantSchemaVersion {
		return s.schemaError("unsupported or missing grepai schema version")
	}
	if int(metadata["embedding_dimensions"].GetIntegerValue()) != s.dimensions {
		return s.schemaError(fmt.Sprintf("embedding dimension mismatch: collection=%d config=%d", metadata["embedding_dimensions"].GetIntegerValue(), s.dimensions))
	}
	if metadata["embedding_cache_namespace"].GetStringValue() != s.embeddingCacheNamespace {
		return s.schemaError("embedding model/cache namespace mismatch")
	}

	vectors := info.Config.Params.VectorsConfig.GetParamsMap()
	if vectors == nil || vectors.Map[denseVectorName] == nil {
		return s.schemaError("named dense vector is missing")
	}
	dense := vectors.Map[denseVectorName]
	if dense.Size != uint64(s.dimensions) || dense.Distance != qdrant.Distance_Cosine {
		return s.schemaError("dense vector configuration mismatch")
	}
	sparse := info.Config.Params.SparseVectorsConfig
	if sparse == nil || sparse.Map[bm25VectorName] == nil || sparse.Map[bm25VectorName].GetModifier() != qdrant.Modifier_Idf {
		return s.schemaError("BM25 sparse vector with IDF is missing")
	}
	if sparse.Map[metadataVectorName] == nil {
		return s.schemaError("metadata sparse vector is missing")
	}
	return nil
}

func (s *QdrantStore) schemaError(reason string) error {
	return fmt.Errorf("Qdrant collection %q is incompatible with grepai schema v%d: %s; choose a new collection name or rebuild the collection", s.collectionName, qdrantSchemaVersion, reason)
}

func (s *QdrantStore) ensurePayloadIndexes(ctx context.Context, info *qdrant.CollectionInfo) error {
	type payloadIndex struct {
		name   string
		prefix bool
	}
	indexes := []payloadIndex{
		{payloadRecordType, false},
		{payloadFilePath, true},
		{payloadContentHash, false},
		{payloadCacheNS, false},
	}
	if s.namespace != "" {
		indexes = append(indexes, payloadIndex{qdrantNamespaceField, false})
	}
	for _, index := range indexes {
		if existing, ok := info.PayloadSchema[index.name]; ok {
			if existing.GetDataType() != qdrant.PayloadSchemaType_Keyword {
				return s.schemaError(fmt.Sprintf("payload index %q has the wrong type", index.name))
			}
			if index.prefix {
				params := existing.GetParams().GetKeywordIndexParams()
				if params == nil || params.Prefix == nil {
					return s.schemaError(fmt.Sprintf("payload index %q does not support prefix matching", index.name))
				}
			}
			continue
		}

		request := &qdrant.CreateFieldIndexCollection{
			CollectionName: s.collectionName,
			FieldName:      index.name,
			Wait:           qdrant.PtrOf(true),
			FieldType:      qdrant.FieldType_FieldTypeKeyword.Enum(),
		}
		if index.prefix {
			request.FieldIndexParams = qdrant.NewPayloadIndexParamsKeyword(
				&qdrant.KeywordIndexParams{Prefix: &qdrant.KeywordPrefixParams{}},
			)
		}
		if _, err := s.client.CreateFieldIndex(ctx, request); err != nil {
			return fmt.Errorf("failed to create Qdrant payload index %q: %w", index.name, err)
		}
	}
	return nil
}

// filter combines conditions with this store's project namespace. It returns
// nil when there is nothing to filter on.
func (s *QdrantStore) filter(conditions ...*qdrant.Condition) *qdrant.Filter {
	if s.namespace != "" {
		conditions = append(conditions, qdrant.NewMatch(qdrantNamespaceField, s.namespace))
	}
	if len(conditions) == 0 {
		return nil
	}
	return &qdrant.Filter{Must: conditions}
}

func (s *QdrantStore) recordTypeFilter(recordType string) *qdrant.Filter {
	return s.filter(qdrant.NewMatch(payloadRecordType, recordType))
}

func sanitizeUTF8(value string) string {
	if utf8.ValidString(value) {
		return value
	}
	return strings.ToValidUTF8(value, "�")
}

// SanitizeCollectionName creates a collision-resistant collection name from a
// repository path.
func SanitizeCollectionName(path string) string {
	canonical := filepath.ToSlash(filepath.Clean(path))
	slugSource := filepath.Base(filepath.Clean(path))
	var slug strings.Builder
	for _, r := range strings.ToLower(slugSource) {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '-' || r == '_' {
			slug.WriteRune(r)
		} else {
			slug.WriteByte('_')
		}
	}
	name := strings.Trim(slug.String(), "_-")
	if name == "" || name == "." {
		name = "repo"
	}
	if len(name) > 48 {
		name = name[:48]
	}
	digest := sha256.Sum256([]byte(canonical))
	return fmt.Sprintf("grepai_%s_%s", name, hex.EncodeToString(digest[:6]))
}

func sanitizeCollectionName(path string) string {
	return SanitizeCollectionName(path)
}

func (s *QdrantStore) pointUUID(kind, id string) uuid.UUID {
	if s.namespace != "" {
		// Chunk IDs and document paths are relative, identical across
		// projects: without the namespace two projects overwrite each other.
		return uuid.NewSHA1(qdrantUUIDNamespace, []byte(kind+":"+s.namespace+"\x00"+id))
	}
	return uuid.NewSHA1(qdrantUUIDNamespace, []byte(kind+":"+id))
}

func (s *QdrantStore) pointID(kind, id string) *qdrant.PointId {
	return qdrant.NewID(s.pointUUID(kind, id).String())
}

func (s *QdrantStore) getUUIDForChunk(chunkID string) uuid.UUID {
	return s.pointUUID(recordChunk, chunkID)
}

func bm25Document(text string) *qdrant.Document {
	return &qdrant.Document{
		Model: bm25ModelName,
		Text:  sanitizeUTF8(text),
		Options: qdrant.NewValueMap(map[string]any{
			"stemmer":       map[string]any{"type": "none"},
			"stopwords":     map[string]any{},
			"tokenizer":     "word",
			"ascii_folding": false,
		}),
	}
}

func (s *QdrantStore) SaveChunks(ctx context.Context, chunks []Chunk) error {
	if len(chunks) == 0 {
		return nil
	}
	batch := s.upsertBatchSize
	if batch <= 0 {
		// A zero batch size would never advance the loop.
		batch = defaultUpsertBatch
	}
	for start := 0; start < len(chunks); start += batch {
		end := start + batch
		if end > len(chunks) {
			end = len(chunks)
		}
		points := make([]*qdrant.PointStruct, 0, end-start)
		for _, chunk := range chunks[start:end] {
			if len(chunk.Vector) != s.dimensions {
				return fmt.Errorf("chunk %q has %d vector dimensions, expected %d", chunk.ID, len(chunk.Vector), s.dimensions)
			}
			payload, err := s.buildChunkPayload(chunk)
			if err != nil {
				return fmt.Errorf("failed to build payload for chunk %q: %w", chunk.ID, err)
			}
			points = append(points, &qdrant.PointStruct{
				Id: s.pointID(recordChunk, chunk.ID),
				Vectors: qdrant.NewVectorsMap(map[string]*qdrant.Vector{
					denseVectorName: qdrant.NewVector(chunk.Vector...),
					bm25VectorName:  qdrant.NewVectorDocument(bm25Document(chunk.Content)),
				}),
				Payload: payload,
			})
		}
		if _, err := s.client.Upsert(ctx, &qdrant.UpsertPoints{
			CollectionName: s.collectionName,
			Wait:           qdrant.PtrOf(true),
			Points:         points,
		}); err != nil {
			return fmt.Errorf("failed to upsert Qdrant chunk batch: %w", err)
		}
	}
	return nil
}

func (s *QdrantStore) buildChunkPayload(chunk Chunk) (map[string]*qdrant.Value, error) {
	fields := map[string]any{
		payloadRecordType:  recordChunk,
		payloadChunkID:     chunk.ID,
		payloadFilePath:    chunk.FilePath,
		payloadStartLine:   chunk.StartLine,
		payloadEndLine:     chunk.EndLine,
		payloadContent:     sanitizeUTF8(chunk.Content),
		payloadHash:        chunk.Hash,
		payloadContentHash: chunk.ContentHash,
		payloadCacheNS:     s.embeddingCacheNamespace,
		payloadUpdatedAt:   chunk.UpdatedAt.UTC().Format(time.RFC3339Nano),
	}
	if chunk.FileHash != "" {
		fields[payloadFileHash] = chunk.FileHash
	}
	if s.namespace != "" {
		fields[qdrantNamespaceField] = s.namespace
	}
	return qdrant.TryValueMap(fields)
}

// DeleteByFile removes the file's chunks together with its document record in
// one operation, so a crash before the new chunks are saved never leaves a
// document that the next scan would trust and skip (#356).
func (s *QdrantStore) DeleteByFile(ctx context.Context, filePath string) error {
	_, err := s.client.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: s.collectionName,
		Wait:           qdrant.PtrOf(true),
		Points:         qdrant.NewPointsSelectorFilter(s.filter(qdrant.NewMatch(payloadFilePath, filePath))),
	})
	if err != nil {
		return fmt.Errorf("failed to delete Qdrant chunks for %q: %w", filePath, err)
	}
	return nil
}

func (s *QdrantStore) searchFilter(opts SearchOptions) *qdrant.Filter {
	filter := s.recordTypeFilter(recordChunk)
	prefixes := normalizedSearchPathPrefixes(opts)
	switch len(prefixes) {
	case 0:
	case 1:
		filter.Must = append(filter.Must, qdrant.NewMatchPrefix(payloadFilePath, prefixes[0]))
	default:
		filter.Should = make([]*qdrant.Condition, 0, len(prefixes))
		for _, prefix := range prefixes {
			filter.Should = append(filter.Should, qdrant.NewMatchPrefix(payloadFilePath, prefix))
		}
	}
	return filter
}

func (s *QdrantStore) Search(ctx context.Context, queryVector []float32, limit int, opts SearchOptions) ([]SearchResult, error) {
	if limit <= 0 {
		return nil, fmt.Errorf("limit must be positive, got: %d", limit)
	}
	if len(queryVector) != s.dimensions {
		return nil, fmt.Errorf("query has %d vector dimensions, expected %d", len(queryVector), s.dimensions)
	}
	points, err := s.client.Query(ctx, &qdrant.QueryPoints{
		CollectionName: s.collectionName,
		Query:          qdrant.NewQuery(queryVector...),
		Using:          qdrant.PtrOf(denseVectorName),
		Filter:         s.searchFilter(opts),
		Limit:          qdrant.PtrOf(uint64(limit)),
		WithPayload:    qdrant.NewWithPayloadInclude(payloadChunkID, payloadFilePath, payloadStartLine, payloadEndLine, payloadContent, payloadHash, payloadContentHash, payloadUpdatedAt),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to search Qdrant: %w", err)
	}
	return parseScoredPoints(points), nil
}

func (s *QdrantStore) SearchHybrid(ctx context.Context, query string, queryVector []float32, limit int, opts SearchOptions, rrfK float32) ([]SearchResult, error) {
	if limit <= 0 {
		return nil, fmt.Errorf("limit must be positive, got: %d", limit)
	}
	if len(queryVector) != s.dimensions {
		return nil, fmt.Errorf("query has %d vector dimensions, expected %d", len(queryVector), s.dimensions)
	}
	if strings.TrimSpace(query) == "" {
		return s.Search(ctx, queryVector, limit, opts)
	}
	k := uint32(math.Round(float64(rrfK)))
	if k == 0 {
		k = 60
	}
	filter := s.searchFilter(opts)
	points, err := s.client.Query(ctx, &qdrant.QueryPoints{
		CollectionName: s.collectionName,
		Prefetch: []*qdrant.PrefetchQuery{
			{Query: qdrant.NewQuery(queryVector...), Using: qdrant.PtrOf(denseVectorName), Filter: filter, Limit: qdrant.PtrOf(uint64(limit))},
			{Query: qdrant.NewQueryNearest(qdrant.NewVectorInputDocument(bm25Document(query))), Using: qdrant.PtrOf(bm25VectorName), Filter: filter, Limit: qdrant.PtrOf(uint64(limit))},
		},
		Query:       qdrant.NewQueryRRF(&qdrant.Rrf{K: qdrant.PtrOf(k)}),
		Filter:      filter,
		Limit:       qdrant.PtrOf(uint64(limit)),
		WithPayload: qdrant.NewWithPayloadInclude(payloadChunkID, payloadFilePath, payloadStartLine, payloadEndLine, payloadContent, payloadHash, payloadContentHash, payloadUpdatedAt),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to run Qdrant hybrid search: %w", err)
	}
	return parseScoredPoints(points), nil
}

func parseScoredPoints(points []*qdrant.ScoredPoint) []SearchResult {
	results := make([]SearchResult, 0, len(points))
	for _, point := range points {
		results = append(results, SearchResult{Chunk: *parseChunkPayload(point.Payload), Score: point.Score})
	}
	return results
}

func parseChunkPayload(payload map[string]*qdrant.Value) *Chunk {
	chunk := &Chunk{}
	chunk.ID = payload[payloadChunkID].GetStringValue()
	chunk.FilePath = payload[payloadFilePath].GetStringValue()
	chunk.StartLine = int(payload[payloadStartLine].GetIntegerValue())
	chunk.EndLine = int(payload[payloadEndLine].GetIntegerValue())
	chunk.Content = payload[payloadContent].GetStringValue()
	chunk.Hash = payload[payloadHash].GetStringValue()
	chunk.FileHash = payload[payloadFileHash].GetStringValue()
	chunk.ContentHash = payload[payloadContentHash].GetStringValue()
	if raw := payload[payloadUpdatedAt].GetStringValue(); raw != "" {
		chunk.UpdatedAt, _ = time.Parse(time.RFC3339Nano, raw)
	}
	return chunk
}

func (s *QdrantStore) parseChunkPayload(payload map[string]*qdrant.Value) *Chunk {
	return parseChunkPayload(payload)
}

func stringsToAny(values []string) []any {
	result := make([]any, len(values))
	for i, value := range values {
		result[i] = value
	}
	return result
}

func payloadStrings(value *qdrant.Value) []string {
	if value == nil || value.GetListValue() == nil {
		return nil
	}
	values := value.GetListValue().GetValues()
	result := make([]string, 0, len(values))
	for _, item := range values {
		result = append(result, item.GetStringValue())
	}
	return result
}

func (s *QdrantStore) GetDocument(ctx context.Context, filePath string) (*Document, error) {
	points, err := s.client.Get(ctx, &qdrant.GetPoints{
		CollectionName: s.collectionName,
		Ids:            []*qdrant.PointId{s.pointID(recordDocument, filePath)},
		WithPayload:    qdrant.NewWithPayload(true),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get Qdrant document %q: %w", filePath, err)
	}
	if len(points) == 0 {
		return nil, nil
	}
	payload := points[0].Payload
	doc := &Document{
		Path:     payload[payloadFilePath].GetStringValue(),
		Hash:     payload[payloadDocumentHash].GetStringValue(),
		ChunkIDs: payloadStrings(payload[payloadChunkIDs]),
	}
	if raw := payload[payloadDocumentMTime].GetStringValue(); raw != "" {
		doc.ModTime, _ = time.Parse(time.RFC3339Nano, raw)
	}
	return doc, nil
}

func (s *QdrantStore) SaveDocument(ctx context.Context, doc Document) error {
	fields := map[string]any{
		payloadRecordType:    recordDocument,
		payloadFilePath:      doc.Path,
		payloadDocumentHash:  doc.Hash,
		payloadDocumentMTime: doc.ModTime.UTC().Format(time.RFC3339Nano),
		payloadChunkIDs:      stringsToAny(doc.ChunkIDs),
		payloadUpdatedAt:     time.Now().UTC().Format(time.RFC3339Nano),
	}
	if s.namespace != "" {
		fields[qdrantNamespaceField] = s.namespace
	}
	payload, err := qdrant.TryValueMap(fields)
	if err != nil {
		return fmt.Errorf("failed to build Qdrant document payload: %w", err)
	}
	_, err = s.client.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: s.collectionName,
		Wait:           qdrant.PtrOf(true),
		Points: []*qdrant.PointStruct{{
			Id: s.pointID(recordDocument, doc.Path),
			Vectors: qdrant.NewVectorsMap(map[string]*qdrant.Vector{
				metadataVectorName: qdrant.NewVectorSparse([]uint32{0}, []float32{1}),
			}),
			Payload: payload,
		}},
	})
	if err != nil {
		return fmt.Errorf("failed to save Qdrant document %q: %w", doc.Path, err)
	}
	return nil
}

func (s *QdrantStore) DeleteDocument(ctx context.Context, filePath string) error {
	_, err := s.client.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: s.collectionName,
		Wait:           qdrant.PtrOf(true),
		Points:         qdrant.NewPointsSelector(s.pointID(recordDocument, filePath)),
	})
	if err != nil {
		return fmt.Errorf("failed to delete Qdrant document %q: %w", filePath, err)
	}
	return nil
}

func (s *QdrantStore) scrollAll(ctx context.Context, filter *qdrant.Filter, payloadFields []string, withVectors *qdrant.WithVectorsSelector, visit func(*qdrant.RetrievedPoint) error) error {
	request := &qdrant.ScrollPoints{
		CollectionName: s.collectionName,
		Filter:         filter,
		Limit:          qdrant.PtrOf(uint32(defaultScrollPage)),
		WithVectors:    withVectors,
	}
	if payloadFields == nil {
		request.WithPayload = qdrant.NewWithPayload(true)
	} else {
		request.WithPayload = qdrant.NewWithPayloadInclude(payloadFields...)
	}
	iterator := s.client.ScrollAll(ctx, request)
	for {
		points, err := iterator.Next()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		for _, point := range points {
			if err := visit(point); err != nil {
				return err
			}
		}
	}
}

func (s *QdrantStore) ListDocuments(ctx context.Context) ([]string, error) {
	paths := make([]string, 0)
	err := s.scrollAll(ctx, s.recordTypeFilter(recordDocument), []string{payloadFilePath}, nil, func(point *qdrant.RetrievedPoint) error {
		paths = append(paths, point.Payload[payloadFilePath].GetStringValue())
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list Qdrant documents: %w", err)
	}
	sort.Strings(paths)
	return paths, nil
}

func (s *QdrantStore) Load(context.Context) error    { return nil }
func (s *QdrantStore) Persist(context.Context) error { return nil }
func (s *QdrantStore) Close() error                  { return s.client.Close() }

func (s *QdrantStore) GetStats(ctx context.Context) (*IndexStats, error) {
	exact := true
	files, err := s.client.Count(ctx, &qdrant.CountPoints{CollectionName: s.collectionName, Filter: s.recordTypeFilter(recordDocument), Exact: &exact})
	if err != nil {
		return nil, fmt.Errorf("failed to count Qdrant documents: %w", err)
	}
	chunks, err := s.client.Count(ctx, &qdrant.CountPoints{CollectionName: s.collectionName, Filter: s.recordTypeFilter(recordChunk), Exact: &exact})
	if err != nil {
		return nil, fmt.Errorf("failed to count Qdrant chunks: %w", err)
	}
	if files > uint64(^uint(0)>>1) || chunks > uint64(^uint(0)>>1) {
		return nil, fmt.Errorf("Qdrant counts exceed maximum int value")
	}
	stats := &IndexStats{TotalFiles: int(files), TotalChunks: int(chunks)}
	err = s.scrollAll(ctx, s.recordTypeFilter(recordDocument), []string{payloadUpdatedAt}, nil, func(point *qdrant.RetrievedPoint) error {
		updated, parseErr := time.Parse(time.RFC3339Nano, point.Payload[payloadUpdatedAt].GetStringValue())
		if parseErr == nil && updated.After(stats.LastUpdated) {
			stats.LastUpdated = updated
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to calculate Qdrant index timestamp: %w", err)
	}
	return stats, nil
}

func (s *QdrantStore) ListFilesWithStats(ctx context.Context) ([]FileStats, error) {
	files := make([]FileStats, 0)
	err := s.scrollAll(ctx, s.recordTypeFilter(recordDocument), []string{payloadFilePath, payloadDocumentMTime, payloadChunkIDs}, nil, func(point *qdrant.RetrievedPoint) error {
		payload := point.Payload
		stat := FileStats{Path: payload[payloadFilePath].GetStringValue(), ChunkCount: len(payloadStrings(payload[payloadChunkIDs]))}
		stat.ModTime, _ = time.Parse(time.RFC3339Nano, payload[payloadDocumentMTime].GetStringValue())
		files = append(files, stat)
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list Qdrant file stats: %w", err)
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })
	return files, nil
}

func (s *QdrantStore) GetChunksForFile(ctx context.Context, filePath string) ([]Chunk, error) {
	filter := s.filter(
		qdrant.NewMatch(payloadRecordType, recordChunk),
		qdrant.NewMatch(payloadFilePath, filePath),
	)
	chunks := make([]Chunk, 0)
	err := s.scrollAll(ctx, filter, nil, qdrant.NewWithVectorsInclude(denseVectorName), func(point *qdrant.RetrievedPoint) error {
		chunk := *parseChunkPayload(point.Payload)
		chunk.Vector = denseVectorFromPoint(point)
		chunks = append(chunks, chunk)
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get Qdrant chunks for %q: %w", filePath, err)
	}
	sort.Slice(chunks, func(i, j int) bool { return chunks[i].StartLine < chunks[j].StartLine })
	return chunks, nil
}

func (s *QdrantStore) GetAllChunks(ctx context.Context) ([]Chunk, error) {
	chunks := make([]Chunk, 0)
	err := s.scrollAll(ctx, s.recordTypeFilter(recordChunk), nil, nil, func(point *qdrant.RetrievedPoint) error {
		chunks = append(chunks, *parseChunkPayload(point.Payload))
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get all Qdrant chunks: %w", err)
	}
	return chunks, nil
}

func denseVectorFromPoint(point *qdrant.RetrievedPoint) []float32 {
	if point == nil || point.Vectors == nil || point.Vectors.GetVectors() == nil {
		return nil
	}
	vector := point.Vectors.GetVectors().GetVectors()[denseVectorName]
	if vector == nil || vector.GetDense() == nil {
		return nil
	}
	return vector.GetDense().GetData()
}

// LookupByContentHash returns the vector of a stored chunk with the same
// content hash whose content starts with contentPrefix (the "File: <path>"
// header the vector was embedded with, #366).
func (s *QdrantStore) LookupByContentHash(ctx context.Context, contentHash, contentPrefix string) ([]float32, bool, error) {
	if contentHash == "" {
		return nil, false, nil
	}
	filter := s.filter(
		qdrant.NewMatch(payloadRecordType, recordChunk),
		qdrant.NewMatch(payloadContentHash, contentHash),
		qdrant.NewMatch(payloadCacheNS, s.embeddingCacheNamespace),
	)
	// Qdrant cannot match a payload prefix without a full-text index: check
	// the content of the points sharing this content hash (usually one or a
	// few). Past qdrantLookupLimit a miss just re-embeds the chunk, which is
	// cheaper than scanning every copy.
	points, err := s.client.Scroll(ctx, &qdrant.ScrollPoints{
		CollectionName: s.collectionName,
		Filter:         filter,
		Limit:          qdrant.PtrOf(uint32(qdrantLookupLimit)),
		WithPayload:    qdrant.NewWithPayloadInclude(payloadContent),
		WithVectors:    qdrant.NewWithVectorsInclude(denseVectorName),
	})
	if err != nil {
		return nil, false, fmt.Errorf("failed to look up Qdrant content hash: %w", err)
	}
	// Stored content is sanitized (see buildChunkPayload): compare likewise.
	contentPrefix = sanitizeUTF8(contentPrefix)
	for _, point := range points {
		if !strings.HasPrefix(point.Payload[payloadContent].GetStringValue(), contentPrefix) {
			continue
		}
		if vector := denseVectorFromPoint(point); len(vector) > 0 {
			return vector, true, nil
		}
	}
	return nil, false, nil
}
