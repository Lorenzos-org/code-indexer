"""ChromaDB wrapper — two collections (code + sessions), full state tracking."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)

# File ID used as a metadata key to group and delete chunks per file
FILE_ID_KEY = "file_id"


class VectorStore:
    def __init__(self, config):
        db_path = Path(config.chroma.persist_dir).expanduser()
        db_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(db_path))

        hnsw_meta = {
            "hnsw:space":           "cosine",
            "hnsw:M":               config.chroma.hnsw_m,
            "hnsw:construction_ef": config.chroma.hnsw_ef_construction,
        }

        self.code_col    = self.client.get_or_create_collection("code",     metadata=hnsw_meta)
        self.session_col = self.client.get_or_create_collection("sessions", metadata=hnsw_meta)

        self._state_path = db_path / "file_state.json"
        self._state: Dict[str, str] = self._load_state()  # {file_path: md5}

    # ── state persistence ──────────────────────────────────────────────────

    def _load_state(self) -> Dict[str, str]:
        if self._state_path.exists():
            try:
                return json.loads(self._state_path.read_text())
            except Exception:
                return {}
        return {}

    def _save_state(self):
        self._state_path.write_text(json.dumps(self._state, indent=2))

    # ── file tracking ──────────────────────────────────────────────────────

    def needs_indexing(self, file_path: str, file_hash: str) -> bool:
        return self._state.get(file_path) != file_hash

    def known_files(self) -> List[str]:
        return list(self._state.keys())

    # ── code collection ────────────────────────────────────────────────────

    def upsert_file(self, chunks, embeddings: List[List[float]], file_hash: str):
        """Remove old chunks for a file and insert the new ones atomically."""
        if not chunks:
            return

        file_path = chunks[0].file_path
        file_id   = file_hash  # use hash as stable group ID

        # Delete previous version
        self._delete_by_file_path(self.code_col, file_path)

        ids       = [f"{file_id}_{i}" for i in range(len(chunks))]
        documents = [c.content for c in chunks]
        metadatas = [{**c.to_metadata(), FILE_ID_KEY: file_id} for c in chunks]

        # Insert in batches of 100 (ChromaDB sweet spot)
        for i in range(0, len(ids), 100):
            self.code_col.add(
                ids=ids[i:i+100],
                documents=documents[i:i+100],
                embeddings=embeddings[i:i+100],
                metadatas=metadatas[i:i+100],
            )

        self._state[file_path] = file_hash
        self._save_state()

    def remove_file(self, file_path: str):
        self._delete_by_file_path(self.code_col, file_path)
        self._state.pop(file_path, None)
        self._save_state()

    def _delete_by_file_path(self, col, file_path: str):
        try:
            existing = col.get(where={"file_path": file_path})
            if existing["ids"]:
                col.delete(ids=existing["ids"])
        except Exception as e:
            logger.debug("Delete (non-fatal): %s", e)

    # ── session collection ─────────────────────────────────────────────────

    def add_session_entry(self, entry_id: str, document: str,
                          embedding: List[float], metadata: Dict):
        self.session_col.add(
            ids=[entry_id],
            documents=[document],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def trim_sessions(self, max_count: int):
        total = self.session_col.count()
        if total <= max_count:
            return
        overflow = total - max_count
        old = self.session_col.get(limit=overflow)
        if old["ids"]:
            self.session_col.delete(ids=old["ids"])

    # ── querying ───────────────────────────────────────────────────────────

    def query_code(self, embedding: List[float], n: int = 10,
                   language: Optional[str] = None,
                   path_contains: Optional[str] = None) -> List[Dict]:
        where = self._build_where(language=language, path_contains=path_contains)
        return self._query(self.code_col, embedding, n, where)

    def query_sessions(self, embedding: List[float], n: int = 10) -> List[Dict]:
        return self._query(self.session_col, embedding, n, where=None)

    def _build_where(self, language=None, path_contains=None) -> Optional[Dict]:
        conditions = []
        if language:
            conditions.append({"language": {"$eq": language}})
        if path_contains:
            conditions.append({"file_path": {"$contains": path_contains}})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _query(col, embedding: List[float], n: int, where: Optional[Dict]) -> List[Dict]:
        kwargs: Dict[str, Any] = dict(
            query_embeddings=[embedding],
            n_results=min(n, max(col.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where
        res = col.query(**kwargs)
        return [
            {
                "content":  doc,
                "metadata": meta,
                "score":    round(1.0 - dist, 4),
            }
            for doc, meta, dist in zip(
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
            )
        ]

    # ── stats ──────────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict:
        return {
            "code_chunks":    self.code_col.count(),
            "session_chunks": self.session_col.count(),
            "indexed_files":  len(self._state),
        }