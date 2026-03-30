"""JSONL session logger — every query and index run is recorded and searchable."""
from __future__ import annotations
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionLogger:
    def __init__(self, config):
        self.log_dir      = Path(config.sessions.log_dir).expanduser()
        self.max_sessions = config.sessions.max_sessions
        self.top_k        = config.sessions.store_top_k_results
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._session_id: Optional[str] = None
        self._session_file: Optional[Path] = None

    # ── session lifecycle ──────────────────────────────────────────────────

    def start_session(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_id   = f"{ts}_{uuid.uuid4().hex[:6]}"
        self._session_file = self.log_dir / f"{self._session_id}.jsonl"
        self._session_file.touch()
        self._cleanup()
        return self._session_id

    @property
    def session_id(self) -> str:
        if not self._session_id:
            self.start_session()
        return self._session_id  # type: ignore[return-value]

    # ── writing ────────────────────────────────────────────────────────────

    def _write(self, event_type: str, data: Dict):
        if not self._session_file:
            self.start_session()
        event = {"type": event_type, "ts": datetime.now().isoformat(), **data}
        with open(self._session_file, "a") as f:  # type: ignore[arg-type]
            f.write(json.dumps(event) + "\n")

    def log_query(self, query: str, results: List[Dict],
                  filters: Dict, duration_ms: int):
        self._write("query", {
            "query":       query,
            "filters":     {k: v for k, v in filters.items() if v},
            "duration_ms": duration_ms,
            "result_count": len(results),
            "top_results": [
                {"score": r["score"], "file": r["metadata"].get("file_path", ""),
                 "lines": f"{r['metadata'].get('start_line','')}–{r['metadata'].get('end_line','')}"}
                for r in results[: self.top_k]
            ],
        })

    def log_index(self, path: str, files_indexed: int, files_skipped: int,
                  chunks_created: int, duration_ms: int, errors: List[str]):
        self._write("index", {
            "path":          path,
            "files_indexed": files_indexed,
            "files_skipped": files_skipped,
            "chunks":        chunks_created,
            "duration_ms":   duration_ms,
            "errors":        errors[:10],  # cap stored errors
        })

    def log_error(self, error: str, context: Optional[Dict] = None):
        self._write("error", {"error": error, "context": context or {}})

    def log_custom(self, tag: str, data: Dict):
        """Extensible: log arbitrary events (e.g. LLM responses, tool calls)."""
        self._write(tag, data)

    # ── reading ────────────────────────────────────────────────────────────

    def get_session(self, session_id: str) -> List[Dict]:
        p = self.log_dir / f"{session_id}.jsonl"
        if not p.exists():
            return []
        events = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return events

    def list_sessions(self, limit: int = 20) -> List[Dict]:
        files = sorted(
            self.log_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]

        sessions = []
        for f in files:
            lines = f.read_text().strip().splitlines()
            if not lines:
                continue
            first = json.loads(lines[0])
            last  = json.loads(lines[-1]) if len(lines) > 1 else first
            sessions.append({
                "id":            f.stem,
                "started":       first.get("ts"),
                "last_activity": last.get("ts"),
                "events":        len(lines),
            })
        return sessions

    def iter_all_events(self, event_type: Optional[str] = None):
        """Yield events across all session files (for external indexing)."""
        for f in sorted(self.log_dir.glob("*.jsonl"),
                        key=lambda p: p.stat().st_mtime):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                    if event_type is None or ev.get("type") == event_type:
                        yield ev
                except json.JSONDecodeError:
                    pass

    # ── cleanup ────────────────────────────────────────────────────────────

    def _cleanup(self):
        files = sorted(
            self.log_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in files[self.max_sessions:]:
            try:
                old.unlink()
            except OSError:
                pass