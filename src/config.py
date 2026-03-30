"""Single source of truth for configuration."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "nomic-embed-text:latest"
    timeout: int = 120
    batch_size: int = 8
    num_threads: int = 2
    inter_batch_delay: float = 0.15


@dataclass
class ChromaConfig:
    persist_dir: str = "~/.local/share/code-indexer/chroma"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 100


@dataclass
class IndexingConfig:
    chunk_lines: int = 60
    chunk_overlap: int = 12
    max_file_kb: int = 500
    respect_gitignore: bool = True
    extensions: List[str] = field(default_factory=lambda: [
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
        ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
        ".kt", ".sh", ".bash", ".sql", ".lua", ".r",
        ".md", ".txt", ".rst", ".html", ".css", ".scss",
        ".yaml", ".yml", ".toml", ".json", ".xml", ".ini", ".env",
    ])
    exclude_dirs: List[str] = field(default_factory=lambda: [
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        "dist", "build", "target", "bin", "obj", ".tox",
        ".mypy_cache", ".pytest_cache", ".next", ".nuxt", ".cache",
        "vendor", ".bundle", "coverage", ".nyc_output",
    ])


@dataclass
class SessionConfig:
    log_dir: str = "~/.local/share/code-indexer/sessions"
    max_sessions: int = 500
    store_top_k_results: int = 5


@dataclass
class DaemonConfig:
    interval_seconds: int = 300
    paths: List[str] = field(default_factory=list)


@dataclass
class Config:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    sessions: SessionConfig = field(default_factory=SessionConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)


def load_config(path: Optional[str] = None) -> Config:
    config_path = Path(path or "config.yaml")
    cfg = Config()

    if not config_path.exists():
        return cfg

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    section_map = {
        "ollama":    (cfg.ollama,    OllamaConfig),
        "chroma":    (cfg.chroma,    ChromaConfig),
        "indexing":  (cfg.indexing,  IndexingConfig),
        "sessions":  (cfg.sessions,  SessionConfig),
        "daemon":    (cfg.daemon,    DaemonConfig),
    }

    for key, (obj, _) in section_map.items():
        if key in data and isinstance(data[key], dict):
            for k, v in data[key].items():
                if hasattr(obj, k):
                    setattr(obj, k, v)

    return cfg