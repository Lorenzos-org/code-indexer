"""Ollama embedding wrapper with in-memory cache and CPU throttle."""
from __future__ import annotations
import hashlib
import logging
import time
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    def __init__(self, config):
        self.base_url = config.ollama.base_url
        self.model = config.ollama.model
        self.timeout = config.ollama.timeout
        self.batch_size = config.ollama.batch_size
        self.num_threads = config.ollama.num_threads
        self.delay = config.ollama.inter_batch_delay

        # In-process cache: avoids re-embedding identical chunks
        self._cache: Dict[str, List[float]] = {}
        self._stats = {"hits": 0, "misses": 0, "errors": 0}

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:20]

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Single HTTP call to Ollama /api/embed (new batch endpoint)."""
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": texts,
                "options": {"num_thread": self.num_threads},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    # ── public API ─────────────────────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        k = self._key(text)
        if k in self._cache:
            self._stats["hits"] += 1
            return self._cache[k]
        self._stats["misses"] += 1
        try:
            emb = self._call_api([text])[0]
            self._cache[k] = emb
            return emb
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("Embed failed: %s", e)
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts in throttled batches, using cache."""
        results: List[List[float]] = []
        uncached_idx: List[int] = []
        uncached_texts: List[str] = []

        # Serve cached first
        for i, t in enumerate(texts):
            k = self._key(t)
            if k in self._cache:
                self._stats["hits"] += 1
                results.append(self._cache[k])
            else:
                results.append([])  # placeholder
                uncached_idx.append(i)
                uncached_texts.append(t)

        # Embed uncached in batches
        for i in range(0, len(uncached_texts), self.batch_size):
            batch = uncached_texts[i : i + self.batch_size]
            try:
                embs = self._call_api(batch)
                for j, emb in enumerate(embs):
                    orig_idx = uncached_idx[i + j]
                    k = self._key(batch[j])
                    self._cache[k] = emb
                    results[orig_idx] = emb
                    self._stats["misses"] += 1
            except Exception as e:
                self._stats["errors"] += 1
                logger.error("Batch embed error: %s", e)
                raise
            time.sleep(self.delay)

        return results

    def clear_cache(self):
        self._cache.clear()

    @property
    def stats(self) -> Dict:
        return {"cache_size": len(self._cache), **self._stats}