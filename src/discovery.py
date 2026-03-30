"""File discovery with gitignore awareness and hash-based change detection."""
from __future__ import annotations
import fnmatch
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Set


@dataclass
class FileDiscovery:
    extensions:       Set[str]
    exclude_dirs:     Set[str]
    respect_gitignore: bool
    max_file_bytes:   int

    def discover(self, root: str) -> Iterator[Path]:
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            return

        gitignore = self._parse_gitignore(root_path / ".gitignore") if self.respect_gitignore else []

        for path in root_path.rglob("*"):
            if not path.is_file():
                continue

            # Extension
            if path.suffix.lower() not in self.extensions:
                continue

            # Excluded dirs (any part of relative path)
            try:
                rel = path.relative_to(root_path)
            except ValueError:
                continue
            if any(part in self.exclude_dirs for part in rel.parts):
                continue

            # Size
            try:
                if path.stat().st_size > self.max_file_bytes:
                    continue
            except OSError:
                continue

            # Gitignore
            if gitignore and self._gitignore_match(rel, gitignore):
                continue

            yield path

    # ── gitignore ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_gitignore(path: Path) -> List[str]:
        if not path.exists():
            return []
        patterns = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
        return patterns

    @staticmethod
    def _gitignore_match(rel: Path, patterns: List[str]) -> bool:
        rel_str = str(rel)
        for pat in patterns:
            if pat.endswith("/"):
                if pat[:-1] in rel.parts:
                    return True
            elif "*" in pat or "?" in pat:
                if fnmatch.fnmatch(rel_str, pat) or fnmatch.fnmatch(rel.name, pat):
                    return True
            else:
                if pat in rel.parts or rel_str == pat or rel.name == pat:
                    return True
        return False

    # ── hashing ────────────────────────────────────────────────────────────

    @staticmethod
    def file_hash(path: str) -> str:
        h = hashlib.md5()
        try:
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(65536), b""):
                    h.update(block)
            return h.hexdigest()
        except OSError:
            return ""