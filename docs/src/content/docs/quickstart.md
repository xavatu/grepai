---
title: Quick Start
description: Index your first project with grepai and run a natural language code search in a few minutes.
---

This page takes you from a fresh machine to your first semantic search. Some steps are done once per machine, the rest once per project.

The whole per-project flow, once grepai and Ollama are set up:

```bash
cd /path/to/your/project
grepai init --provider ollama --backend gob --yes   # create .grepai/config.yaml
grepai watch --background                           # index now and keep the index current
grepai status --no-ui                               # check the index
grepai search "error handling middleware"
```

The steps below explain each command and the options you have.

## Before Your First Project

These steps are done once per machine. Every project then shares the grepai binary, Ollama and the embedding model.

Install grepai from this fork (Go 1.25 or newer and a C compiler are required, because tree-sitter uses cgo):

```bash
go install github.com/xavatu/grepai/cmd/grepai@main
```

See [Installation from this fork](https://github.com/xavatu/grepai#from-this-fork) to build from a clone instead. The Homebrew formula and install scripts on the [Installation](/grepai/installation/) page install upstream builds, which lack the [Qdrant hybrid search and reranker](#optional-qdrant-hybrid-search-and-reranker) of this fork.

The default setup embeds code locally with Ollama and the `nomic-embed-text` model. If Ollama does not run as a service, start it in a separate terminal and leave that terminal open:

```bash
ollama serve
```

In another terminal, download the model and check both tools:

```bash
ollama pull nomic-embed-text
grepai version
ollama list          # must list nomic-embed-text
```

## 1. Initialize the Project

From the root of your project, run:

```bash
cd /path/to/your/project
grepai init --provider ollama --backend gob --yes
```

This picks Ollama and a local GOB file without prompts. `grepai init --yes` gives the same result, since these are the defaults. Other ways to run it:

```bash
grepai init                            # answer each prompt, Enter accepts the default
grepai init --ui                       # interactive wizard in the terminal
grepai init -p openai -b gob --yes     # another provider, set with flags
```

This creates `.grepai/config.yaml`. If the project has a `.gitignore`, grepai adds `.grepai/` to it. After initialization, grepai commands work from any subdirectory of the project. See [Configuration](/grepai/configuration/) to change settings later, and [grepai init](/grepai/commands/grepai_init/) for all flags.

## 2. Build and Maintain the Index

For day-to-day use, run the watcher in the background:

```bash
grepai watch --background
```

The watcher does two things:

1. It runs an initial scan: it splits each supported file into chunks, embeds them, and stores the vectors. Files that have not changed since the last run are skipped.
2. It keeps running and re-indexes files as you create, edit, rename or delete them.

`--background` waits up to 30 seconds for the initial scan. A small project is fully indexed when the command returns. On a larger project the command reports that initial indexing is still in progress and returns, while the watcher keeps indexing. In both cases it prints the log file of the watcher.

If the watcher cannot start, the error names its log file. `cannot connect to Ollama` in the log means that Ollama is not running at the configured endpoint: start `ollama serve` and run `grepai watch --background` again.

A missing model does not stop the watcher. Each file fails with `model "nomic-embed-text" not found` in the log, and the index stays empty. Run `ollama pull nomic-embed-text`, then restart the watcher with `grepai watch --stop` and `grepai watch --background`: its initial scan indexes the files again.

To follow indexing in the terminal instead, run the watcher in the foreground:

```bash
grepai watch          # live dashboard in an interactive terminal
grepai watch --no-ui  # plain log output, for scripts or CI
```

**Don't need a daemon?** `grepai index` runs the same scan once and exits. Run it again whenever you want to refresh the index:

```bash
grepai index
```

See the [File Watching guide](/grepai/watch-guide/) for details.

## 3. Check the Index

```bash
grepai status --no-ui
```

A ready index reports its files and chunks:

```
$ grepai status --no-ui
grepai index status
Files indexed: 2
Total chunks: 2
Index size: 10.6 KB
Last updated: 2026-10-25 12:29:27
Provider: ollama (nomic-embed-text)
Watcher: not running
Watcher log: /Users/you/Library/Logs/grepai/grepai-worktree-3890120fa89c.log
```

`no grepai project found` means you are outside an initialized project: go to the project, or run `grepai init` first. While the background watcher still indexes, the counts grow from one run to the next. If `Files indexed` stays at 0, look for embedding errors in the watcher log (see step 2).

Without `--no-ui`, in an interactive terminal, `grepai status` opens a browser where you can list indexed files and inspect their chunks.

## 4. Search Your Code

From the project root or any of its subdirectories:

```bash
# Describe what the code does
grepai search "user authentication"
grepai search "how errors are handled"

# Return 5 results instead of the default 10
grepai search "database queries" --limit 5

# Only search under a path
grepai search "REST API routes" --path internal/api

# JSON output for scripts and AI agents (--compact drops the code content)
grepai search "authentication" --json --compact
```

See [grepai search](/grepai/commands/grepai_search/) for all flags, including the token-efficient `--toon` format.

### Example Output

```
$ grepai search "error handling middleware" --limit 2
Found 2 results for: "error handling middleware"

─── Result 1 (score: 0.6864) ───
File: middleware/error.go:1-15

   1 │ package middleware
   2 │
   3 │ import "net/http"
   4 │
   5 │ // ErrorHandler recovers from panics and returns a 500 response.
   6 │ func ErrorHandler(next http.Handler) http.Handler {
   7 │ 	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
   ...

─── Result 2 (score: 0.5687) ───
File: handlers/user.go:1-11

   1 │ package handlers
   2 │
   3 │ import "errors"
   ...
```

Each result shows its rank, its similarity score, the file and line range, and up to 15 lines of the chunk. Scores are relative: compare them within one search rather than across searches.

With `--json --compact`, you get only the location and score:

```json
[
  {
    "file_path": "middleware/error.go",
    "start_line": 1,
    "end_line": 15,
    "score": 0.68637824
  }
]
```

## 5. Connect to Your AI Agent

Your AI agent does not know about the index yet. The recommended way to connect it is **[MCP](/grepai/mcp/)**: `grepai mcp-serve` exposes search, trace and index status as native tools for any MCP-compatible agent.

If your agent does not support MCP, run `grepai agent-setup`. It appends instructions for using the grepai CLI to the agent files it finds in your project (`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`, `.cursorrules`, `.cursor/rules`, `.windsurfrules`). You do not need it if you use MCP.

Claude Code users can also add the optional **[subagent](/grepai/subagent/)** (`grepai agent-setup --with-subagent`), which gives exploration agents grepai access in their own context.

## 6. Manage the Background Watcher

Run these commands from the project the watcher belongs to:

```bash
grepai watch --status   # is it running, and where is its log
grepai watch --stop     # stop it
```

Stopping is graceful: the watcher saves the index before it exits. Stop it when you no longer work on the project. When you start it again with `grepai watch --background`, the initial scan only re-indexes the files that changed in between.

## 7. Use grepai in Another Project

Each project has its own index and its own watcher. For another project, repeat only the project steps:

```bash
cd /path/to/another/project
grepai init --provider ollama --backend gob --yes
grepai watch --background
grepai status --no-ui
```

## Optional: Qdrant Hybrid Search and Reranker

On large repositories, this fork can move the index to [Qdrant](/grepai/backends/stores/#qdrant) and add two search stages. **Hybrid search** combines vector search with BM25 keyword ranking inside Qdrant. The **reranker** sends the best candidates to a neural model served by llama-server, which reorders them before grepai returns the results.

Start the services from a clone of this fork. Qdrant 1.19 or newer is required. The reranker profile needs an NVIDIA GPU with the NVIDIA Container Toolkit, and its first start downloads about 3.3 GB:

```bash
git clone https://github.com/xavatu/grepai.git
cd grepai
docker compose --profile qdrant --profile reranker up -d
curl http://127.0.0.1:8081/health   # the reranker answers once the model is loaded
```

Then initialize the project with Qdrant:

```bash
cd /path/to/your/project
grepai init --provider ollama --backend qdrant --yes
```

In `.grepai/config.yaml`, switch on the two stages. `grepai init` already wrote both keys:

```yaml
search:
  hybrid:
    enabled: true     # dense vectors and BM25, fused inside Qdrant
reranker:
  enabled: true       # llama-server at http://127.0.0.1:8081
```

Start indexing with `grepai watch --background` as above. If the reranker is unavailable at search time, grepai logs a warning and returns results in retrieval order.

To move a project that is already indexed, stop its watcher and set `store.backend: qdrant` in its config. The initial scan of the next watcher fills the new collection. See [Vector Stores](/grepai/backends/stores/#qdrant), [Hybrid Search](/grepai/hybrid-search/) and [Neural Reranking](/grepai/configuration/#neural-reranking-disabled-by-default) for all settings.

## Tips for Better Searches

- **Be descriptive**: "user login validation" works better than "login".
- **Describe intent**: say what the code does, not how it is written ("where are users saved" rather than "func saveUser").
- **Write queries in English**: the default model, `nomic-embed-text`, is trained mainly on English.
- **Keep grep for exact matches**: to find a known identifier or string, `grep` or `ripgrep` is still the right tool.

## Next Steps

- [Semantic Search Guide](/grepai/search-guide/): write better queries
- [File Watching Guide](/grepai/watch-guide/): how the daemon keeps the index fresh
- [Call Graph Analysis](/grepai/trace/): find callers and callees with `grepai trace`
- [Configuration](/grepai/configuration/): change the embedder, storage, chunking and ignore rules
- [Commands Reference](/grepai/commands/grepai/): every command and flag
