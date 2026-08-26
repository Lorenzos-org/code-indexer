"""Unit tests for CodeIndexer - Main orchestrator component."""
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from src.indexer import CodeIndexer
from src.config import Config, OllamaConfig, ChromaConfig, IndexingConfig, SessionConfig


class TestCodeIndexerInitialization:
    """Tests for CodeIndexer initialization."""

    def test_init_creates_components(self):
        """Verify all components are initialized on construction."""
        config = Config(
            ollama=OllamaConfig(model="nomic-embed-text"),
            chroma=ChromaConfig(persist_dir="./chroma_test"),
            indexing=IndexingConfig(extensions=[".py"], exclude_dirs=["build"]),
            sessions=SessionConfig(log_dir="./sessions_test")
        )
        
        with patch('src.indexer.OllamaEmbedder') as MockEmbedder, \
             patch('src.indexer.CodeChunker') as MockChunker, \
             patch('src.indexer.FileDiscovery') as MockDiscovery, \
             patch('src.indexer.VectorStore') as MockStore, \
             patch('src.indexer.SessionLogger') as MockSessions:
            
            indexer = CodeIndexer(config)
        
        MockEmbedder.assert_called_once_with(config)
        MockChunker.assert_called_once_with(config)
        MockStore.assert_called_once_with(config)
        MockSessions.assert_called_once_with(config)
        
        assert indexer.embedder is not None
        assert indexer.chunker is not None
        assert indexer.discovery is not None
        assert indexer.store is not None
        assert indexer.sessions is not None

    def test_init_discovery_uses_config_values(self):
        """Verify FileDiscovery receives correct config values."""
        config = Config(
            ollama=OllamaConfig(),
            chroma=ChromaConfig(persist_dir="./chroma"),
            indexing=IndexingConfig(
                extensions=[".py", ".js"],
                exclude_dirs=["node_modules", "venv"],
                respect_gitignore=True,
                max_file_kb=500
            ),
            sessions=SessionConfig()
        )
        
        with patch('src.indexer.OllamaEmbedder'), \
             patch('src.indexer.CodeChunker'), \
             patch('src.indexer.VectorStore'), \
             patch('src.indexer.SessionLogger'), \
             patch('src.indexer.FileDiscovery') as MockDiscovery:
            
            CodeIndexer(config)
        
        MockDiscovery.assert_called_once_with(
            extensions={".py", ".js"},
            exclude_dirs={"node_modules", "venv"},
            respect_gitignore=True,
            max_file_bytes=500 * 1024
        )


class TestIndexMethod:
    """Tests for the index() method."""

    @pytest.fixture
    def mock_indexer(self):
        """Create a CodeIndexer with all dependencies mocked."""
        config = Config(
            ollama=OllamaConfig(),
            chroma=ChromaConfig(persist_dir="./chroma"),
            indexing=IndexingConfig(),
            sessions=SessionConfig()
        )
        
        indexer = CodeIndexer.__new__(CodeIndexer)
        indexer.config = config
        indexer.embedder = MagicMock()
        indexer.chunker = MagicMock()
        indexer.discovery = MagicMock()
        indexer.store = MagicMock()
        indexer.sessions = MagicMock()
        
        # Set default return values
        from src.discovery import FileDiscovery
        patcher = patch.object(FileDiscovery, 'file_hash', return_value='default_hash')
        patcher.start()
        
        return indexer

    def test_index_discovers_files(self, mock_indexer):
        """Verify index discovers files from the given path."""
        mock_indexer.discovery.discover.return_value = [
            Path("/test/file1.py"),
            Path("/test/file2.py")
        ]
        mock_indexer.store.needs_indexing.return_value = True
        mock_indexer.chunker.chunk_file.return_value = []
        
        mock_indexer.index("/test/path")
        
        mock_indexer.discovery.discover.assert_called_once_with("/test/path")

    def test_index_skips_unchanged_files(self, mock_indexer):
        """Verify unchanged files are skipped in incremental mode."""
        mock_indexer.discovery.discover.return_value = [Path("/test/file.py")]
        mock_indexer.store.needs_indexing.return_value = False
        
        stats = mock_indexer.index("/test/path", incremental=True)
        
        assert stats["files_scanned"] == 1
        assert stats["files_skipped"] == 1
        assert stats["files_indexed"] == 0
        mock_indexer.chunker.chunk_file.assert_not_called()

    def test_index_processes_changed_files(self, mock_indexer, sample_python_code):
        """Verify changed files are processed and indexed."""
        from src.chunker import Chunk
        
        mock_indexer.discovery.discover.return_value = [Path("/test/file.py")]
        mock_indexer.store.needs_indexing.return_value = True
        mock_indexer.chunker.chunk_file.return_value = [
            Chunk(content=sample_python_code[:50], file_path="/test/file.py", 
                  start_line=1, end_line=3, language="python")
        ]
        mock_indexer.embedder.embed_batch.return_value = [[0.1] * 384]
        
        stats = mock_indexer.index("/test/path", incremental=True)
        
        assert stats["files_scanned"] == 1
        assert stats["files_indexed"] == 1
        assert stats["chunks_created"] == 1
        mock_indexer.store.upsert_file.assert_called_once()

    def test_index_handles_missing_hash(self, mock_indexer):
        """Verify files without hash are skipped."""
        mock_indexer.discovery.discover.return_value = [Path("/test/file.py")]
        
        with patch('src.indexer.FileDiscovery.file_hash', return_value=None):
            stats = mock_indexer.index("/test/path")
        
        assert stats["files_scanned"] == 1
        # Files with missing hash are silently skipped (not counted in files_skipped)
        assert stats["files_indexed"] == 0
        assert stats["chunks_created"] == 0

    def test_index_records_errors(self, mock_indexer):
        """Verify errors during indexing are captured."""
        mock_indexer.discovery.discover.return_value = [Path("/test/file.py")]
        mock_indexer.store.needs_indexing.return_value = True
        mock_indexer.chunker.chunk_file.side_effect = Exception("Parse error")
        
        stats = mock_indexer.index("/test/path")
        
        assert len(stats["errors"]) == 1
        assert "Parse error" in stats["errors"][0]

    def test_index_returns_duration(self, mock_indexer):
        """Verify index returns execution duration."""
        mock_indexer.discovery.discover.return_value = []
        
        stats = mock_indexer.index("/test/path")
        
        assert "duration_ms" in stats
        assert isinstance(stats["duration_ms"], int)

    def test_index_prunes_on_incremental(self, mock_indexer):
        """Verify _prune_deleted is called in incremental mode."""
        mock_indexer.discovery.discover.return_value = []
        
        with patch.object(mock_indexer, '_prune_deleted') as mock_prune:
            mock_indexer.index("/test/path", incremental=True)
            mock_prune.assert_called_once_with("/test/path")

    def test_index_does_not_prune_on_full(self, mock_indexer):
        """Verify _prune_deleted is NOT called in full mode."""
        mock_indexer.discovery.discover.return_value = []
        
        with patch.object(mock_indexer, '_prune_deleted') as mock_prune:
            mock_indexer.index("/test/path", incremental=False)
            mock_prune.assert_not_called()

    def test_index_logs_session(self, mock_indexer):
        """Verify indexing session is logged."""
        mock_indexer.discovery.discover.return_value = []
        
        mock_indexer.index("/test/path")
        
        mock_indexer.sessions.log_index.assert_called_once()
        call_args = mock_indexer.sessions.log_index.call_args
        assert call_args[1]["path"] == "/test/path"


class TestIndexFileMethod:
    """Tests for index_file() method."""

    @pytest.fixture
    def mock_indexer(self):
        """Create a CodeIndexer with mocked dependencies."""
        config = Config(
            ollama=OllamaConfig(),
            chroma=ChromaConfig(persist_dir="./chroma"),
            indexing=IndexingConfig(),
            sessions=SessionConfig()
        )
        
        with patch('src.indexer.OllamaEmbedder'), \
             patch('src.indexer.CodeChunker'), \
             patch('src.indexer.VectorStore'), \
             patch('src.indexer.SessionLogger'), \
             patch('src.indexer.FileDiscovery'):
            
            indexer = CodeIndexer(config)
            return indexer

    def test_index_file_single_file(self, mock_indexer, sample_python_code):
        """Verify single file indexing works correctly."""
        from src.chunker import Chunk
        
        with patch('src.indexer.FileDiscovery.file_hash', return_value="abc123"):
            mock_indexer.chunker.chunk_file.return_value = [
                Chunk(content=sample_python_code[:50], file_path="/test.py",
                      start_line=1, end_line=2, language="python")
            ]
            mock_indexer.embedder.embed_batch.return_value = [[0.1] * 384]
            
            mock_indexer.index_file("/test.py")
        
        mock_indexer.store.upsert_file.assert_called_once()

    def test_index_file_handles_errors(self, mock_indexer):
        """Verify errors in index_file are logged."""
        with patch('src.indexer.FileDiscovery.file_hash', side_effect=Exception("IO Error")):
            mock_indexer.index_file("/test.py")
        
        mock_indexer.sessions.log_error.assert_called_once()


class TestRemoveFileMethod:
    """Tests for remove_file() method."""

    @pytest.fixture
    def mock_indexer(self):
        """Create a CodeIndexer with mocked dependencies."""
        config = Config(
            ollama=OllamaConfig(),
            chroma=ChromaConfig(persist_dir="./chroma"),
            indexing=IndexingConfig(),
            sessions=SessionConfig()
        )
        
        with patch('src.indexer.OllamaEmbedder'), \
             patch('src.indexer.CodeChunker'), \
             patch('src.indexer.VectorStore'), \
             patch('src.indexer.SessionLogger'), \
             patch('src.indexer.FileDiscovery'):
            
            indexer = CodeIndexer(config)
            return indexer

    def test_remove_file_calls_store(self, mock_indexer):
        """Verify remove_file delegates to store."""
        mock_indexer.remove_file("/test/file.py")
        
        mock_indexer.store.remove_file.assert_called_once()


class TestPruneDeletedMethod:
    """Tests for _prune_deleted() method."""

    @pytest.fixture
    def mock_indexer(self):
        """Create a CodeIndexer with mocked dependencies."""
        config = Config(
            ollama=OllamaConfig(),
            chroma=ChromaConfig(persist_dir="./chroma"),
            indexing=IndexingConfig(),
            sessions=SessionConfig()
        )
        
        with patch('src.indexer.OllamaEmbedder'), \
             patch('src.indexer.CodeChunker'), \
             patch('src.indexer.VectorStore'), \
             patch('src.indexer.SessionLogger'), \
             patch('src.indexer.FileDiscovery'):
            
            indexer = CodeIndexer(config)
            return indexer

    def test_prune_removes_nonexistent_files(self, mock_indexer):
        """Verify prune removes files that no longer exist."""
        mock_indexer.store.known_files.return_value = [
            "/test/root/existing/file.py",
            "/test/root/deleted/file.py"
        ]
        
        # Mock Path.exists to return False for deleted file only
        original_exists = Path.exists
        def mock_exists(self):
            return "deleted" not in str(self)
        
        Path.exists = mock_exists
        try:
            mock_indexer._prune_deleted("/test/root")
        finally:
            Path.exists = original_exists
        
        mock_indexer.store.remove_file.assert_called_once_with("/test/root/deleted/file.py")

    def test_prune_removes_outside_root(self, mock_indexer):
        """Verify prune removes files outside the root path."""
        mock_indexer.store.known_files.return_value = [
            "/test/root/inside/file.py",
            "/other/path/outside/file.py"
        ]
        
        # Both files exist, but one is outside the root
        original_exists = Path.exists
        Path.exists = lambda self: True
        try:
            mock_indexer._prune_deleted("/test/root")
        finally:
            Path.exists = original_exists
        
        # Should only remove the outside file
        calls = mock_indexer.store.remove_file.call_args_list
        assert len(calls) == 1
        assert calls[0][0][0] == "/other/path/outside/file.py"


class TestQueryCodeMethod:
    """Tests for query_code() method."""

    @pytest.fixture
    def mock_indexer(self):
        """Create a CodeIndexer with mocked dependencies."""
        config = Config(
            ollama=OllamaConfig(),
            chroma=ChromaConfig(persist_dir="./chroma"),
            indexing=IndexingConfig(),
            sessions=SessionConfig()
        )
        
        with patch('src.indexer.OllamaEmbedder'), \
             patch('src.indexer.CodeChunker'), \
             patch('src.indexer.VectorStore'), \
             patch('src.indexer.SessionLogger'), \
             patch('src.indexer.FileDiscovery'):
            
            indexer = CodeIndexer(config)
            return indexer

    def test_query_code_embeds_and_searches(self, mock_indexer):
        """Verify query_code embeds text and searches store."""
        mock_indexer.embedder.embed.return_value = [0.1] * 384
        mock_indexer.store.query_code.return_value = [
            {"content": "result", "metadata": {}, "score": 0.9}
        ]
        
        result = mock_indexer.query_code("test query")
        
        mock_indexer.embedder.embed.assert_called_once_with("test query")
        mock_indexer.store.query_code.assert_called_once()
        assert "results" in result
        assert "duration_ms" in result

    def test_query_code_passes_filters(self, mock_indexer):
        """Verify query_code passes filter parameters."""
        mock_indexer.embedder.embed.return_value = [0.1] * 384
        mock_indexer.store.query_code.return_value = []
        
        mock_indexer.query_code("query", n=10, language="python", path_contains="utils")
        
        mock_indexer.store.query_code.assert_called_once_with(
            [0.1] * 384, n=10, language="python", path_contains="utils"
        )

    def test_query_code_logs_session(self, mock_indexer):
        """Verify query_code logs the session."""
        mock_indexer.embedder.embed.return_value = [0.1] * 384
        mock_indexer.store.query_code.return_value = []
        
        mock_indexer.query_code("test query")
        
        mock_indexer.sessions.log_query.assert_called_once()


class TestQuerySessionsMethod:
    """Tests for query_sessions() method."""

    @pytest.fixture
    def mock_indexer(self):
        """Create a CodeIndexer with mocked dependencies."""
        config = Config(
            ollama=OllamaConfig(),
            chroma=ChromaConfig(persist_dir="./chroma"),
            indexing=IndexingConfig(),
            sessions=SessionConfig()
        )
        
        with patch('src.indexer.OllamaEmbedder'), \
             patch('src.indexer.CodeChunker'), \
             patch('src.indexer.VectorStore'), \
             patch('src.indexer.SessionLogger'), \
             patch('src.indexer.FileDiscovery'):
            
            indexer = CodeIndexer(config)
            return indexer

    def test_query_sessions_embeds_and_searches(self, mock_indexer):
        """Verify query_sessions embeds and queries session collection."""
        mock_indexer.embedder.embed.return_value = [0.1] * 384
        mock_indexer.store.query_sessions.return_value = [
            {"content": "session result", "metadata": {}, "score": 0.85}
        ]
        
        result = mock_indexer.query_sessions("find my session")
        
        mock_indexer.embedder.embed.assert_called_once_with("find my session")
        mock_indexer.store.query_sessions.assert_called_once()
        assert len(result["results"]) == 1


class TestLogConversationMethod:
    """Tests for log_conversation() method."""

    @pytest.fixture
    def mock_indexer(self):
        """Create a CodeIndexer with mocked dependencies."""
        config = Config(
            ollama=OllamaConfig(),
            chroma=ChromaConfig(persist_dir="./chroma"),
            indexing=IndexingConfig(),
            sessions=SessionConfig()
        )
        
        with patch('src.indexer.OllamaEmbedder'), \
             patch('src.indexer.CodeChunker'), \
             patch('src.indexer.VectorStore'), \
             patch('src.indexer.SessionLogger') as MockSessionLogger, \
             patch('src.indexer.FileDiscovery'):
            
            indexer = CodeIndexer(config)
            indexer.sessions = MockSessionLogger.return_value
            indexer.sessions.session_id = "test-session-123"
            return indexer

    def test_log_conversation_creates_entry(self, mock_indexer):
        """Verify log_conversation creates a session entry."""
        mock_indexer.embedder.embed.return_value = [0.5] * 384
        
        mock_indexer.log_conversation("What is this?", "This is an answer", tag="qa")
        
        mock_indexer.embedder.embed.assert_called_once()
        mock_indexer.store.add_session_entry.assert_called_once()
        
        call_args = mock_indexer.store.add_session_entry.call_args
        assert "Q: What is this?\nA: This is an answer" in call_args[1]["document"]
        assert call_args[1]["metadata"]["tag"] == "qa"

    def test_log_conversation_trims_sessions(self, mock_indexer):
        """Verify log_conversation trims old sessions."""
        mock_indexer.embedder.embed.return_value = [0.5] * 384
        mock_indexer.config.sessions.max_sessions = 100
        
        mock_indexer.log_conversation("Q", "A")
        
        mock_indexer.store.trim_sessions.assert_called_once_with(1000)


class TestStatsProperty:
    """Tests for stats property."""

    @pytest.fixture
    def mock_indexer(self):
        """Create a CodeIndexer with mocked dependencies."""
        config = Config(
            ollama=OllamaConfig(),
            chroma=ChromaConfig(persist_dir="./chroma"),
            indexing=IndexingConfig(),
            sessions=SessionConfig()
        )
        
        with patch('src.indexer.OllamaEmbedder'), \
             patch('src.indexer.CodeChunker'), \
             patch('src.indexer.VectorStore'), \
             patch('src.indexer.SessionLogger'), \
             patch('src.indexer.FileDiscovery'):
            
            indexer = CodeIndexer(config)
            return indexer

    def test_stats_aggregates_components(self, mock_indexer):
        """Verify stats aggregates data from components."""
        mock_indexer.store.stats = {"code_chunks": 100, "session_chunks": 20}
        mock_indexer.embedder.stats = {"cache_hits": 50, "calls": 100}
        
        stats = mock_indexer.stats
        
        assert stats["store"]["code_chunks"] == 100
        assert stats["embedder"]["cache_hits"] == 50
