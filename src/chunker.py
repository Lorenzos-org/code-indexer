"""Code-aware chunker: smart block-boundary splitting + context extraction."""
from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ── Language mapping ────────────────────────────────────────────────────────

EXTENSION_LANGUAGE: Dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "jsx", ".tsx": "tsx", ".java": "java",
    ".go": "go", ".rs": "rust", ".c": "c", ".cpp": "cpp",
    ".h": "c_header", ".hpp": "cpp_header", ".cs": "csharp",
    ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".kt": "kotlin", ".scala": "scala", ".sh": "shell",
    ".bash": "shell", ".sql": "sql", ".html": "html",
    ".css": "css", ".scss": "scss", ".md": "markdown",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".xml": "xml", ".r": "r", ".lua": "lua",
}

# Regex patterns marking the start of a top-level block in each language
BLOCK_STARTERS: Dict[str, List[str]] = {
    "python":     [r"^(\s*)class\s+", r"^(\s*)def\s+", r"^(\s*)async\s+def\s+"],
    "javascript": [r"^(\s*)(?:async\s+)?function\s+\w+", r"^(\s*)(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\("],
    "typescript": [r"^(\s*)(?:async\s+)?function\s+\w+", r"^(\s*)(?:const|let|var)\s+\w+\s*[=:]\s*(?:async\s*)?\(", r"^(\s*)(?:export\s+)?(?:abstract\s+)?class\s+"],
    "java":       [r"^(\s*)(?:public|private|protected)?\s*(?:static\s+)?(?:class|interface|enum)\s+", r"^(\s*)(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\]]+\s+\w+\s*\("],
    "go":         [r"^(\s*)func\s+", r"^(\s*)type\s+\w+\s+struct\s*\{"],
    "rust":       [r"^(\s*)(?:pub\s+)?fn\s+", r"^(\s*)(?:pub\s+)?struct\s+", r"^(\s*)impl\s+"],
    "csharp":     [r"^(\s*)(?:public|private|protected|internal)?\s*(?:static\s+)?(?:class|interface|enum|struct)\s+", r"^(\s*)(?:public|private|protected|internal)?\s*(?:static\s+)?(?:async\s+)?[\w<>\[\]]+\s+\w+\s*\("],
    "ruby":       [r"^(\s*)def\s+", r"^(\s*)class\s+", r"^(\s*)module\s+"],
    "kotlin":     [r"^(\s*)(?:fun\s+)", r"^(\s*)(?:data\s+)?class\s+"],
}

# Context extraction patterns per language
CONTEXT_PATTERNS: Dict[str, List[str]] = {
    "python":     [r"(?:async\s+)?def\s+(\w+)", r"class\s+(\w+)"],
    "javascript": [r"(?:async\s+)?function\s+(\w+)", r"(?:const|let|var)\s+(\w+)\s*="],
    "typescript": [r"(?:async\s+)?function\s+(\w+)", r"class\s+(\w+)", r"(?:const|let|var)\s+(\w+)\s*[=:]"],
    "go":         [r"func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)"],
    "rust":       [r"fn\s+(\w+)", r"struct\s+(\w+)", r"impl\s+(\w+)"],
    "java":       [r"class\s+(\w+)", r"interface\s+(\w+)", r"(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\("],
    "csharp":     [r"class\s+(\w+)", r"(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\("],
    "ruby":       [r"def\s+(\w+)", r"class\s+(\w+)", r"module\s+(\w+)"],
    "kotlin":     [r"fun\s+(\w+)", r"class\s+(\w+)"],
}


@dataclass
class Chunk:
    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    content_hash: str = ""
    enclosing_context: str = ""
    line_count: int = 0

    def __post_init__(self):
        self.content_hash = hashlib.md5(self.content.encode()).hexdigest()
        self.line_count = self.end_line - self.start_line

    def to_metadata(self) -> Dict:
        return {
            "file_path":          self.file_path,
            "start_line":         self.start_line,
            "end_line":           self.end_line,
            "language":           self.language,
            "content_hash":       self.content_hash,
            "enclosing_context":  self.enclosing_context,
            "line_count":         self.line_count,
        }


class CodeChunker:
    def __init__(self, config):
        self.chunk_lines  = config.indexing.chunk_lines
        self.overlap      = config.indexing.chunk_overlap
        self.max_file_kb  = config.indexing.max_file_kb * 1024

    # ── public ─────────────────────────────────────────────────────────────

    def chunk_file(self, file_path: str) -> List[Chunk]:
        p = Path(file_path)
        try:
            if p.stat().st_size > self.max_file_kb:
                return []
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        lines = text.splitlines(keepends=True)
        if not lines:
            return []

        lang = EXTENSION_LANGUAGE.get(p.suffix.lower(), "unknown")

        if lang in BLOCK_STARTERS:
            return self._smart_chunk(lines, lang, file_path)
        return self._simple_chunk(lines, lang, file_path)

    # ── strategies ─────────────────────────────────────────────────────────

    def _simple_chunk(self, lines: List[str], lang: str, path: str) -> List[Chunk]:
        chunks = []
        i = 0
        step = max(1, self.chunk_lines - self.overlap)

        while i < len(lines):
            end = min(i + self.chunk_lines, len(lines))
            content = "".join(lines[i:end]).strip()
            if content:
                ctx = self._extract_context(lines[i:end], lang)
                chunks.append(Chunk(
                    content=content, file_path=path,
                    start_line=i + 1, end_line=end,
                    language=lang, enclosing_context=ctx,
                ))
            i += step

        return chunks

    def _smart_chunk(self, lines: List[str], lang: str, path: str) -> List[Chunk]:
        """Chunk at block boundaries; fall back to hard limit if blocks are huge."""
        patterns = [re.compile(p, re.MULTILINE) for p in BLOCK_STARTERS[lang]]
        block_starts = {i for i, line in enumerate(lines) if any(p.match(line) for p in patterns)}

        chunks: List[Chunk] = []
        current: List[str] = []
        current_start = 0
        half = self.chunk_lines // 2

        for i, line in enumerate(lines):
            if i in block_starts and current and (i - current_start) >= half:
                content = "".join(current).strip()
                if content:
                    ctx = self._extract_context(current, lang)
                    chunks.append(Chunk(
                        content=content, file_path=path,
                        start_line=current_start + 1, end_line=i,
                        language=lang, enclosing_context=ctx,
                    ))
                # Keep overlap
                keep = current[-self.overlap:] if len(current) > self.overlap else current[:]
                current = keep
                current_start = i - len(current)

            current.append(line)

            # Hard cap — split regardless of boundaries
            if len(current) >= int(self.chunk_lines * 1.5):
                content = "".join(current).strip()
                if content:
                    ctx = self._extract_context(current, lang)
                    chunks.append(Chunk(
                        content=content, file_path=path,
                        start_line=current_start + 1, end_line=i + 1,
                        language=lang, enclosing_context=ctx,
                    ))
                current = current[-self.overlap:]
                current_start = i + 1 - len(current)

        # Tail
        if current:
            content = "".join(current).strip()
            if content:
                ctx = self._extract_context(current, lang)
                chunks.append(Chunk(
                    content=content, file_path=path,
                    start_line=current_start + 1, end_line=len(lines),
                    language=lang, enclosing_context=ctx,
                ))

        return chunks

    # ── context ────────────────────────────────────────────────────────────

    def _extract_context(self, lines: List[str], lang: str) -> str:
        """Return the nearest enclosing function/class name visible in the chunk header."""
        patterns = CONTEXT_PATTERNS.get(lang, [])
        compiled = [re.compile(p) for p in patterns]

        for line in lines[:15]:
            line = line.strip()
            for pat in compiled:
                m = pat.search(line)
                if m:
                    return m.group(1)
        return ""