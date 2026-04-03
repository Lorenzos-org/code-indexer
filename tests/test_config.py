"""Unit tests for the config module."""
import pytest
import tempfile
from pathlib import Path
from src.config import (
    Config, OllamaConfig, ChromaConfig, IndexingConfig,
    SessionConfig, DaemonConfig, load_config
)


class TestOllamaConfig:
    """Tests for OllamaConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        cfg = OllamaConfig()
        assert cfg.base_url == "http://localhost:11434"
        assert cfg.model == "nomic-embed-text:latest"
        assert cfg.timeout == 120
        assert cfg.batch_size == 8
        assert cfg.num_threads == 2
        assert cfg.inter_batch_delay == 0.15

    def test_custom_values(self):
        """Test that custom values can be set."""
        cfg = OllamaConfig(
            base_url="http://custom:11434",
            model="custom-model",
            timeout=60,
            batch_size=16,
            num_threads=4,
            inter_batch_delay=0.5
        )
        assert cfg.base_url == "http://custom:11434"
        assert cfg.model == "custom-model"
        assert cfg.timeout == 60
        assert cfg.batch_size == 16
        assert cfg.num_threads == 4
        assert cfg.inter_batch_delay == 0.5


class TestChromaConfig:
    """Tests for ChromaConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        cfg = ChromaConfig()
        assert cfg.persist_dir == "~/.local/share/code-indexer/chroma"
        assert cfg.hnsw_m == 16
        assert cfg.hnsw_ef_construction == 100

    def test_custom_values(self):
        """Test that custom values can be set."""
        cfg = ChromaConfig(
            persist_dir="/custom/path",
            hnsw_m=32,
            hnsw_ef_construction=200
        )
        assert cfg.persist_dir == "/custom/path"
        assert cfg.hnsw_m == 32
        assert cfg.hnsw_ef_construction == 200


class TestIndexingConfig:
    """Tests for IndexingConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        cfg = IndexingConfig()
        assert cfg.chunk_lines == 60
        assert cfg.chunk_overlap == 12
        assert cfg.max_file_kb == 500
        assert cfg.respect_gitignore is True
        assert ".py" in cfg.extensions
        assert ".git" in cfg.exclude_dirs

    def test_extensions_list(self):
        """Test that extensions list contains expected values."""
        cfg = IndexingConfig()
        expected_extensions = [
            ".py", ".js", ".ts", ".java", ".go", ".rs",
            ".md", ".yaml", ".json"
        ]
        for ext in expected_extensions:
            assert ext in cfg.extensions

    def test_exclude_dirs_list(self):
        """Test that exclude_dirs list contains expected values."""
        cfg = IndexingConfig()
        expected_dirs = [
            ".git", "__pycache__", "node_modules",
            "venv", "dist", "build"
        ]
        for dir_name in expected_dirs:
            assert dir_name in cfg.exclude_dirs


class TestSessionConfig:
    """Tests for SessionConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        cfg = SessionConfig()
        assert cfg.log_dir == "~/.local/share/code-indexer/sessions"
        assert cfg.max_sessions == 500
        assert cfg.store_top_k_results == 5


class TestDaemonConfig:
    """Tests for DaemonConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        cfg = DaemonConfig()
        assert cfg.interval_seconds == 300
        assert cfg.paths == []

    def test_custom_paths(self):
        """Test that custom paths can be set."""
        cfg = DaemonConfig(interval_seconds=600, paths=["/path1", "/path2"])
        assert cfg.interval_seconds == 600
        assert cfg.paths == ["/path1", "/path2"]


class TestConfig:
    """Tests for main Config dataclass."""

    def test_default_nested_configs(self):
        """Test that Config contains nested configs with defaults."""
        cfg = Config()
        assert isinstance(cfg.ollama, OllamaConfig)
        assert isinstance(cfg.chroma, ChromaConfig)
        assert isinstance(cfg.indexing, IndexingConfig)
        assert isinstance(cfg.sessions, SessionConfig)
        assert isinstance(cfg.daemon, DaemonConfig)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_nonexistent_config(self):
        """Test loading a non-existent config returns defaults."""
        cfg = load_config("/nonexistent/path/config.yaml")
        assert isinstance(cfg, Config)
        assert cfg.ollama.base_url == "http://localhost:11434"

    def test_load_empty_config(self):
        """Test loading an empty config file returns defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name
        try:
            cfg = load_config(temp_path)
            assert isinstance(cfg, Config)
            assert cfg.ollama.model == "nomic-embed-text:latest"
        finally:
            Path(temp_path).unlink()

    def test_load_partial_config(self):
        """Test loading a config with partial overrides."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
ollama:
  base_url: http://override:11434
  model: custom-model
indexing:
  chunk_lines: 100
""")
            temp_path = f.name
        try:
            cfg = load_config(temp_path)
            assert cfg.ollama.base_url == "http://override:11434"
            assert cfg.ollama.model == "custom-model"
            assert cfg.ollama.timeout == 120  # default
            assert cfg.indexing.chunk_lines == 100
            assert cfg.indexing.chunk_overlap == 12  # default
        finally:
            Path(temp_path).unlink()

    def test_load_full_config(self):
        """Test loading a complete config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
ollama:
  base_url: http://test:11434
  model: test-model
  timeout: 30
  batch_size: 4
  num_threads: 1
  inter_batch_delay: 0.1
chroma:
  persist_dir: /test/chroma
  hnsw_m: 8
  hnsw_ef_construction: 50
indexing:
  chunk_lines: 80
  chunk_overlap: 20
  max_file_kb: 1000
  respect_gitignore: false
sessions:
  log_dir: /test/sessions
  max_sessions: 100
  store_top_k_results: 10
daemon:
  interval_seconds: 600
  paths:
    - /path1
    - /path2
""")
            temp_path = f.name
        try:
            cfg = load_config(temp_path)
            assert cfg.ollama.base_url == "http://test:11434"
            assert cfg.ollama.model == "test-model"
            assert cfg.ollama.timeout == 30
            assert cfg.chroma.persist_dir == "/test/chroma"
            assert cfg.indexing.chunk_lines == 80
            assert cfg.indexing.respect_gitignore is False
            assert cfg.sessions.max_sessions == 100
            assert cfg.daemon.interval_seconds == 600
            assert cfg.daemon.paths == ["/path1", "/path2"]
        finally:
            Path(temp_path).unlink()

    def test_load_default_config(self):
        """Test loading when no path specified uses config.yaml."""
        # This will use the existing config.yaml or return defaults
        cfg = load_config()
        assert isinstance(cfg, Config)
