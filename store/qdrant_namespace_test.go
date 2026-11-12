package store

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/qdrant/go-client/qdrant"
	"google.golang.org/protobuf/proto"
)

func TestQdrantProjectNamespace(t *testing.T) {
	ns := func(collection, root string) string {
		t.Helper()
		return QdrantProjectNamespace(collection, root)
	}
	if got := ns("", "/work/a"); got != "" {
		t.Fatalf("derived collection got namespace %q, want none", got)
	}
	tempDir := func() string {
		dir, err := filepath.EvalSymlinks(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		return dir
	}
	plain := tempDir()
	if got := ns("shared", plain); got != plain {
		t.Fatalf("explicit collection outside git got namespace %q, want the project root %q", got, plain)
	}
	// A symlinked root (as an MCP root parameter may be) names the same
	// project as its resolved path, which is what the CLI uses.
	link := filepath.Join(t.TempDir(), "link")
	if err := os.Symlink(plain, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if got := ns("shared", link+string(filepath.Separator)); got != plain {
		t.Fatalf("symlinked root got namespace %q, want %q", got, plain)
	}
	// Without git on PATH, the path is used (git's absence is not fatal).
	t.Run("git not installed", func(t *testing.T) {
		t.Setenv("PATH", "")
		dir := tempDir()
		if got := ns("shared", dir); got != dir {
			t.Fatalf("namespace without git = %q, want %q", got, dir)
		}
	})
	// In a repository with an origin remote, the namespace is the repository,
	// whatever the checkout path, and holds no credential.
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed")
	}
	repo := tempDir()
	for _, args := range [][]string{{"init", "-q"}, {"remote", "add", "origin", "https://token@github.com/Org/App.git"}} {
		if out, err := exec.Command("git", append([]string{"-C", repo}, args...)...).CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	if got := ns("shared", repo); got != "github.com/Org/App" {
		t.Fatalf("git project got namespace %q, want github.com/Org/App", got)
	}
	if got := ns("", repo); got != "" {
		t.Fatalf("derived collection got namespace %q", got)
	}
}

// Collections without a namespace keep the point IDs and payloads they had
// before namespaces existed, so existing indexes are reused as they are.
func TestQdrantUnnamespacedLayoutUnchanged(t *testing.T) {
	plain := &QdrantStore{}
	want := uuid.NewSHA1(qdrantUUIDNamespace, []byte("chunk:README.md_0"))
	if got := plain.getUUIDForChunk("README.md_0"); got != want {
		t.Fatalf("point ID changed: got %v, want %v", got, want)
	}
	payload, err := plain.buildChunkPayload(Chunk{FilePath: "README.md"})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := payload[qdrantNamespaceField]; ok {
		t.Fatal("payload gained a project namespace")
	}
	if f := plain.filter(); f != nil {
		t.Fatalf("unfiltered read got filter %v", f)
	}
	want1 := &qdrant.Filter{Must: []*qdrant.Condition{qdrant.NewMatch(payloadFilePath, "README.md")}}
	if f := plain.filter(qdrant.NewMatch(payloadFilePath, "README.md")); !proto.Equal(f, want1) {
		t.Fatalf("filter = %v, want %v", f, want1)
	}
}

func TestQdrantNamespacedLayout(t *testing.T) {
	a := &QdrantStore{namespace: "/work/a"}
	b := &QdrantStore{namespace: "/work/b"}
	plain := &QdrantStore{}
	if a.getUUIDForChunk("README.md_0") == b.getUUIDForChunk("README.md_0") {
		t.Fatal("two projects got the same point ID for README.md_0")
	}
	if a.getUUIDForChunk("README.md_0") == plain.getUUIDForChunk("README.md_0") {
		t.Fatal("namespaced point ID collides with the unnamespaced one")
	}
	if a.pointUUID(recordDocument, "README.md") == b.pointUUID(recordDocument, "README.md") {
		t.Fatal("two projects got the same document point ID")
	}
	payload, err := a.buildChunkPayload(Chunk{FilePath: "README.md"})
	if err != nil {
		t.Fatal(err)
	}
	if got := payload[qdrantNamespaceField].GetStringValue(); got != "/work/a" {
		t.Fatalf("payload namespace = %q", got)
	}
	ns := qdrant.NewMatch(qdrantNamespaceField, "/work/a")
	if f, want := a.filter(), (&qdrant.Filter{Must: []*qdrant.Condition{ns}}); !proto.Equal(f, want) {
		t.Fatalf("filter = %v, want %v", f, want)
	}
	search := a.searchFilter(SearchOptions{PathPrefix: "src/"})
	want := &qdrant.Filter{Must: []*qdrant.Condition{
		qdrant.NewMatch(payloadRecordType, recordChunk),
		ns,
		qdrant.NewMatchPrefix(payloadFilePath, "src/"),
	}}
	if !proto.Equal(search, want) {
		t.Fatalf("search filter = %v, want %v", search, want)
	}
}

// qdrantTestClient connects to a local Qdrant (gRPC port from
// GREPAI_QDRANT_TEST_GRPC_PORT, 6334 by default) or skips the test.
func qdrantTestClient(t *testing.T) (*qdrant.Client, int) {
	t.Helper()
	port := 6334
	if v := os.Getenv("GREPAI_QDRANT_TEST_GRPC_PORT"); v != "" {
		p, err := strconv.Atoi(v)
		if err != nil {
			t.Fatal(err)
		}
		port = p
	}
	conn, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), time.Second)
	if err != nil {
		t.Skipf("local Qdrant unavailable: %v", err)
	}
	_ = conn.Close()
	client, err := qdrant.NewClient(&qdrant.Config{Host: "127.0.0.1", Port: port})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return client, port
}

// TestQdrantSharedCollectionIsolation reproduces #247 on a real Qdrant: two
// projects indexing the same explicit collection must not see, overwrite or
// delete each other's points or reuse each other's vectors.
func TestQdrantSharedCollectionIsolation(t *testing.T) {
	client, port := qdrantTestClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	collection := fmt.Sprintf("test_grepai_247_%d", time.Now().UnixNano())
	t.Cleanup(func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		_ = client.DeleteCollection(cleanupCtx, collection)
	})
	open := func(namespace string) *QdrantStore {
		t.Helper()
		s, err := NewQdrantStore(ctx, "localhost", port, false, collection, "", 3, WithQdrantProjectNamespace(namespace))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = s.Close() })
		return s
	}
	save := func(s *QdrantStore, path, content string, vec ...float32) {
		t.Helper()
		if err := s.DeleteByFile(ctx, path); err != nil {
			t.Fatal(err)
		}
		chunk := Chunk{ID: path + "_0", FilePath: path, Content: "File: " + path + "\n\n" + content, Hash: content, ContentHash: "ch-" + content, FileHash: "fh-" + content, Vector: vec, UpdatedAt: time.Now()}
		if err := s.SaveChunks(ctx, []Chunk{chunk}); err != nil {
			t.Fatal(err)
		}
		if err := s.SaveDocument(ctx, Document{Path: path, Hash: "fh-" + content, ModTime: time.Now(), ChunkIDs: []string{chunk.ID}}); err != nil {
			t.Fatal(err)
		}
	}
	docs := func(s *QdrantStore) []string {
		t.Helper()
		paths, err := s.ListDocuments(ctx)
		if err != nil {
			t.Fatal(err)
		}
		sort.Strings(paths)
		return paths
	}
	contents := func(s *QdrantStore, path string) []string {
		t.Helper()
		chunks, err := s.GetChunksForFile(ctx, path)
		if err != nil {
			t.Fatal(err)
		}
		var out []string
		for _, c := range chunks {
			out = append(out, c.Content)
		}
		return out
	}

	a := open("/work/a")
	b := open("/work/b")
	save(a, "README.md", "victim", 1, 0, 0)
	save(a, "victim_only.go", "secret", 0, 1, 0)
	save(b, "README.md", "attacker", 1, 1, 0)
	if err := b.DeleteByFile(ctx, "victim_only.go"); err != nil { // B never had it
		t.Fatal(err)
	}

	if got := contents(a, "README.md"); len(got) != 1 || got[0] != "File: README.md\n\nvictim" {
		t.Fatalf("project A README.md = %v", got)
	}
	if got := contents(b, "README.md"); len(got) != 1 || got[0] != "File: README.md\n\nattacker" {
		t.Fatalf("project B README.md = %v", got)
	}
	if got := docs(a); fmt.Sprint(got) != "[README.md victim_only.go]" {
		t.Fatalf("project A documents = %v", got)
	}
	if got := docs(b); fmt.Sprint(got) != "[README.md]" {
		t.Fatalf("project B documents = %v", got)
	}
	if doc, err := b.GetDocument(ctx, "victim_only.go"); err != nil || doc != nil {
		t.Fatalf("project B sees project A's document: %+v, %v", doc, err)
	}
	results, err := b.SearchHybrid(ctx, "victim secret attacker", []float32{1, 0, 0}, 10, SearchOptions{}, 60)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Chunk.FilePath != "README.md" || results[0].Chunk.Hash != "attacker" {
		t.Fatalf("project B hybrid search returned %+v", results)
	}
	stats, err := a.GetStats(ctx)
	if err != nil || stats.TotalChunks != 2 || stats.TotalFiles != 2 {
		t.Fatalf("project A stats = %+v, %v", stats, err)
	}
	if _, found, _ := b.LookupByContentHash(ctx, "ch-victim", "File: README.md\n\n"); found {
		t.Fatal("project B reused project A's vector")
	}
	if vec, found, err := a.LookupByContentHash(ctx, "ch-victim", "File: README.md\n\n"); err != nil || !found || len(vec) != 3 || vec[0] != 1 {
		t.Fatalf("project A vector not reused: vec=%v found=%v err=%v", vec, found, err)
	}
	if _, found, _ := a.LookupByContentHash(ctx, "ch-victim", "File: moved.md\n\n"); found {
		t.Fatal("vector embedded under another file path was reused (#366)")
	}

	// Deleting a file removes its document with its chunks (#356).
	if err := a.DeleteByFile(ctx, "victim_only.go"); err != nil {
		t.Fatal(err)
	}
	if doc, err := a.GetDocument(ctx, "victim_only.go"); err != nil || doc != nil {
		t.Fatalf("document left after DeleteByFile: %+v, %v", doc, err)
	}
}

// A collection created before project namespaces existed has unnamespaced
// points: a namespaced store must keep reading them instead of hiding them.
func TestQdrantPreNamespaceCollectionIgnoresNamespace(t *testing.T) {
	client, port := qdrantTestClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	collection := fmt.Sprintf("test_grepai_prens_%d", time.Now().UnixNano())
	t.Cleanup(func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		_ = client.DeleteCollection(cleanupCtx, collection)
	})
	old, err := NewQdrantStore(ctx, "localhost", port, false, collection, "", 3)
	if err != nil {
		t.Fatal(err)
	}
	defer old.Close()
	if err := client.UpdateCollection(ctx, &qdrant.UpdateCollection{
		CollectionName: collection,
		Metadata:       qdrant.NewValueMap(map[string]any{metadataProjectNamespaces: false}),
	}); err != nil {
		t.Fatal(err)
	}
	if err := old.SaveDocument(ctx, Document{Path: "README.md", Hash: "h", ChunkIDs: []string{"README.md_0"}}); err != nil {
		t.Fatal(err)
	}

	upgraded, err := NewQdrantStore(ctx, "localhost", port, false, collection, "", 3, WithQdrantProjectNamespace("/work/a"))
	if err != nil {
		t.Fatal(err)
	}
	defer upgraded.Close()
	if upgraded.namespace != "" {
		t.Fatalf("namespace = %q on a pre-namespace collection, want none", upgraded.namespace)
	}
	if doc, err := upgraded.GetDocument(ctx, "README.md"); err != nil || doc == nil {
		t.Fatalf("existing document hidden: %+v, %v", doc, err)
	}
}
