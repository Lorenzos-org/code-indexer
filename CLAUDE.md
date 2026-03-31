# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A local semantic code search tool using Ollama for embeddings and ChromaDB for vector storage. It indexes source code files, chunks them intelligently by language, and enables natural language search over your codebase.

## Running the Tool

```bash
# Install dependencies
pip install -r requirements.txt

# Run CLI (requires Ollama running locally)
python -m code_indexer <command>
```

## CLI Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `index PATH` | Index a directory | `python -m code_indexer index ~/myproject --watch` |
| `query TEXT` | Semantic search | `python -m code_indexer query "authentication middleware" --lang python -n 10` |
| `daemon PATHS...` | Background polling | `python -m code_indexer daemon ~/proj1 ~/proj2 --interval 300` |
| `stats` | Index statistics | `python -m code_indexer stats` |
| `session --list` | Browse sessions | `python -m code_indexer session --list --limit 20` |
| `log QUERY RESPONSE` | Log conversation | `python -m code_indexer log "what is X" "X is Y" --tag faq` |

## External Dependencies

- **Ollama**: Must be running at `OLLAMA_BASE_URL` (default: http://localhost:11434)
- **Model**: Uses `nomic-embed-text:latest` by default (configurable)
- **ChromaDB**: Persisted to `~/.local/share/code-indexer/chroma/`

## Architecture Overview

The system has a pipeline architecture orchestrated by `CodeIndexer`:

```
FileDiscovery → CodeChunker → OllamaEmbedder → VectorStore
                     ↓
              SessionLogger (parallel audit trail)
```

### Component Responsibilities

- **`CodeIndexer`** (`src/indexer.py`): Main orchestrator tying all components together. Handles indexing, querying, and session logging.

- **`FileDiscovery`** (`src/discovery.py`): Recursively discovers files matching configured extensions, respects `.gitignore`, filters by size/excluded dirs. Computes MD5 hashes for change detection.

- **`CodeChunker`** (`src/chunker.py`): Language-aware chunking with two strategies:
  - **Smart chunking**: Uses regex patterns (`BLOCK_STARTERS`) to split at function/class boundaries for supported languages (Python, JS/TS, Go, Rust, Java, etc.)
  - **Simple chunking**: Fixed-size sliding window for other files
  Extracts enclosing context (function/class names) for richer metadata.

- **`OllamaEmbedder`** (`src/embedder.py`): HTTP client for Ollama's `/api/embed` endpoint. Features:
  - In-memory SHA256 cache to avoid re-embedding identical chunks
  - Batched embedding with configurable batch size
  - Throttling delay between batches

- **`VectorStore`** (`src/vectorstore.py`): ChromaDB wrapper managing two collections:
  - `code`: Stores code chunks with file metadata
  - `sessions`: Stores conversation/query history
  Maintains `file_state.json` for incremental indexing (path → MD5 hash).

- **`SessionLogger`** (`src/sessions.py`): JSONL audit logger. Each session gets a timestamped file with all queries, index operations, and errors.

### Data Flow

**Indexing**:
1. `FileDiscovery.discover()` yields eligible files
2. Compare file hash against `VectorStore._state` to skip unchanged files
3. `CodeChunker.chunk_file()` splits into chunks with metadata
4. `OllamaEmbedder.embed_batch()` generates embeddings (cached)
5. `VectorStore.upsert_file()` deletes old chunks, inserts new ones, updates state
6. `SessionLogger.log_index()` records operation

**Querying**:
1. Embed query text via `OllamaEmbedder.embed()`
2. `VectorStore.query_code()` performs cosine similarity search with optional filters (language, path substring)
3. `SessionLogger.log_query()` records results
4. Results scored as `1.0 - distance` (cosine similarity)

## Configuration

`config.yaml` at project root (or path via `-c/--config`):

```yaml
ollama:
  base_url: "http://localhost:11434"
  model: "nomic-embed-text:latest"
  batch_size: 8

indexing:
  chunk_lines: 60
  chunk_overlap: 12
  max_file_kb: 500
  extensions: [.py, .js, .ts, ...]
  exclude_dirs: [.git, node_modules, ...]

chroma:
  persist_dir: "~/.local/share/code-indexer/chroma"
```

Configuration loads via `src/config.py` dataclasses with defaults.

## Key Files

- **`__main__.py`**: Entry point, delegates to `src/cli.py`
- **`src/cli.py`**: Argument parsing and command dispatch
- **`config.yaml`**: Default configuration with sensible defaults

## Testing

No test suite exists currently. To run the tool during development:

```bash
# Quick sanity check
python -m code_indexer index . --full
python -m code_indexer query "configuration loading"
python -m code_indexer stats
```
