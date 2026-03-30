"""Main orchestrator — ties all components together."""
from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config
from .chunker import CodeChunker
from .discovery import FileDiscovery
from .embedder import OllamaEmbedder
from .sessions import SessionLogger
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


class CodeIndexer:
    def __init__(self, config: Config):
        self.config    = config
        self.embedder  = OllamaEmbedder(config)
        self.chunker   = CodeChunker(config)
        self.discovery = FileDiscovery(
            extensions=set(config.indexing.extensions),
            exclude_dirs=set(config.indexing.exclude_dirs),
            respect_gitignore=config.indexing.respect_gitignore,
            max_file_bytes=config.indexing.max_file_kb * 1024,
        )
        self.store    = VectorStore(config)
        self.sessions = SessionLogger(config)

    # ── indexing ───────────────────────────────────────────────────────────

    def index(self, path: str, incremental: bool = True) -> Dict:
        """Index all eligible files under `path`."""
        start = time.time()
        stats = dict(files_scanned=0, files_indexed=0,
                     files_skipped=0, chunks_created=0, errors=[])

        # Remove stale entries for files that no longer exist
        if incremental:
            self._prune_deleted(path)

        for file_path in self.discovery.discover(path):
            stats["files_scanned"] += 1
            fp = str(file_path)

            try:
                fhash = FileDiscovery.file_hash(fp)
                if not fhash:
                    continue

                if incremental and not self.store.needs_indexing(fp, fhash):
                    stats["files_skipped"] += 1
                    continue

                chunks = self.chunker.chunk_file(fp)
                if not chunks:
                    stats["files_skipped"] += 1
                    continue

                embeddings = self.embedder.embed_batch([c.content for c in chunks])
                self.store.upsert_file(chunks, embeddings, fhash)

                stats["files_indexed"] += 1
                stats["chunks_created"] += len(chunks)
                logger.debug("Indexed %s — %d chunks", fp, len(chunks))

            except Exception as e:
                msg = f"{fp}: {str(e)[:120]}"
                stats["errors"].append(msg)
                logger.error("Error indexing %s: %s", fp, e)

        duration_ms = int((time.time() - start) * 1000)
        stats["duration_ms"] = duration_ms

        self.sessions.log_index(
            path=path,
            files_indexed=stats["files_indexed"],
            files_skipped=stats["files_skipped"],
            chunks_created=stats["chunks_created"],
            duration_ms=duration_ms,
            errors=stats["errors"],
        )
        return stats

    def index_file(self, file_path: str):
        """Index a single file (used by the file watcher)."""
        fp = str(Path(file_path).resolve())
        try:
            fhash = FileDiscovery.file_hash(fp)
            if not fhash:
                return
            chunks = self.chunker.chunk_file(fp)
            if not chunks:
                return
            embeddings = self.embedder.embed_batch([c.content for c in chunks])
            self.store.upsert_file(chunks, embeddings, fhash)
            logger.debug("Watcher indexed %s — %d chunks", fp, len(chunks))
        except Exception as e:
            logger.error("Watcher error %s: %s", fp, e)
            self.sessions.log_error(str(e), {"file": fp})

    def remove_file(self, file_path: str):
        self.store.remove_file(str(Path(file_path).resolve()))

    def _prune_deleted(self, root_path: str):
        root = str(Path(root_path).expanduser().resolve())
        for fp in self.store.known_files():
            p = Path(fp)
            if not p.exists() or not str(p).startswith(root):
                logger.info("Pruning deleted: %s", fp)
                self.store.remove_file(fp)

    # ── querying ───────────────────────────────────────────────────────────

    def query_code(self, text: str, n: int = 10,
                   language: Optional[str] = None,
                   path_contains: Optional[str] = None) -> Dict:
        start = time.time()
        emb = self.embedder.embed(text)
        results = self.store.query_code(emb, n=n, language=language, path_contains=path_contains)
        duration_ms = int((time.time() - start) * 1000)

        self.sessions.log_query(text, results,
                                {"language": language, "path_contains": path_contains},
                                duration_ms)
        return {"query": text, "results": results, "duration_ms": duration_ms,
                "session_id": self.sessions.session_id}

    def query_sessions(self, text: str, n: int = 10) -> Dict:
        start = time.time()
        emb = self.embedder.embed(text)
        results = self.store.query_sessions(emb, n=n)
        duration_ms = int((time.time() - start) * 1000)
        return {"query": text, "results": results, "duration_ms": duration_ms}

    def log_conversation(self, query: str, response: str,
                         tag: Optional[str] = None):
        """
        Index a Q&A pair into the sessions collection.
        Call this from any LLM wrapper to make past conversations searchable.
        """
        doc  = f"Q: {query}\nA: {response}"
        emb  = self.embedder.embed(doc)
        ts   = str(time.time())
        meta = {
            "type":       "conversation",
            "session_id": self.sessions.session_id,
            "ts":         ts,
            "tag":        tag or "",
        }
        self.store.add_session_entry(
            entry_id=f"{self.sessions.session_id}::{ts}",
            document=doc,
            embedding=emb,
            metadata=meta,
        )
        self.store.trim_sessions(self.config.sessions.max_sessions * 10)

    # ── stats ──────────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict:
        return {
            "store":   self.store.stats,
            "embedder": self.embedder.stats,
        }