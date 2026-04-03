"""Unit tests for VectorStore - ChromaDB wrapper with state tracking."""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.vectorstore import VectorStore, FILE_ID_KEY
from src.config import Config, ChromaConfig


class TestVectorStoreInitialization:
    """Tests for VectorStore initialization and setup."""

    def test_init_creates_persist_directory(self, tmp_path):
        """Verify persist directory is created on init."""
        persist_dir = tmp_path / "chroma_test"
        config = Config(
            chroma=ChromaConfig(persist_dir=str(persist_dir))
        )
        
        with patch('src.vectorstore.chromadb.PersistentClient') as mock_client:
            store = VectorStore(config)
        
        assert persist_dir.exists()
        assert persist_dir.is_dir()

    def test_init_creates_chroma_client(self, tmp_path):
        """Verify ChromaDB client is initialized with correct path."""
        persist_dir = tmp_path / "chroma_test"
        config = Config(
            chroma=ChromaConfig(persist_dir=str(persist_dir))
        )
        
        with patch('src.vectorstore.chromadb.PersistentClient') as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            
            mock_code_col = MagicMock()
            mock_session_col = MagicMock()
            mock_instance.get_or_create_collection.side_effect = [mock_code_col, mock_session_col]
            
            store = VectorStore(config)
        
        MockClient.assert_called_once_with(path=str(persist_dir))
        assert mock_instance.get_or_create_collection.call_count == 2

    def test_init_creates_collections(self, tmp_path):
        """Verify code and sessions collections are created."""
        persist_dir = tmp_path / "chroma_test"
        config = Config(
            chroma=ChromaConfig(persist_dir=str(persist_dir), hnsw_m=16, hnsw_ef_construction=100)
        )
        
        with patch('src.vectorstore.chromadb.PersistentClient') as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            
            mock_code_col = MagicMock()
            mock_session_col = MagicMock()
            mock_instance.get_or_create_collection.side_effect = [mock_code_col, mock_session_col]
            
            store = VectorStore(config)
        
        # Check HNSW metadata
        expected_metadata = {
            "hnsw:space": "cosine",
            "hnsw:M": 16,
            "hnsw:construction_ef": 100,
        }
        
        calls = mock_instance.get_or_create_collection.call_args_list
        assert calls[0][1]["metadata"] == expected_metadata
        assert calls[0][0][0] == "code"
        assert calls[1][0][0] == "sessions"

    def test_init_loads_existing_state(self, tmp_path):
        """Verify existing state file is loaded on init."""
        persist_dir = tmp_path / "chroma_test"
        persist_dir.mkdir()
        state_file = persist_dir / "file_state.json"
        
        existing_state = {"/path/to/file.py": "abc123", "/path/to/other.py": "def456"}
        state_file.write_text(json.dumps(existing_state))
        
        config = Config(
            chroma=ChromaConfig(persist_dir=str(persist_dir))
        )
        
        with patch('src.vectorstore.chromadb.PersistentClient') as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            mock_instance.get_or_create_collection.side_effect = [MagicMock(), MagicMock()]
            
            store = VectorStore(config)
        
        assert store._state == existing_state

    def test_init_handles_corrupt_state_file(self, tmp_path):
        """Verify corrupt state file is handled gracefully."""
        persist_dir = tmp_path / "chroma_test"
        persist_dir.mkdir()
        state_file = persist_dir / "file_state.json"
        
        state_file.write_text("{ invalid json }")
        
        config = Config(
            chroma=ChromaConfig(persist_dir=str(persist_dir))
        )
        
        with patch('src.vectorstore.chromadb.PersistentClient') as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            mock_instance.get_or_create_collection.side_effect = [MagicMock(), MagicMock()]
            
            store = VectorStore(config)
        
        assert store._state == {}


class TestFileTracking:
    """Tests for file state tracking functionality."""

    @pytest.fixture
    def vector_store(self, tmp_path):
        """Create a VectorStore instance with mocked ChromaDB."""
        persist_dir = tmp_path / "chroma_test"
        persist_dir.mkdir()
        
        config = Config(
            chroma=ChromaConfig(persist_dir=str(persist_dir))
        )
        
        with patch('src.vectorstore.chromadb.PersistentClient') as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            mock_instance.get_or_create_collection.side_effect = [MagicMock(), MagicMock()]
            
            store = VectorStore(config)
        
        return store

    def test_needs_indexing_new_file(self, vector_store):
        """Verify new files need indexing."""
        assert vector_store.needs_indexing("/new/file.py", "hash123") is True

    def test_needs_indexing_changed_file(self, vector_store):
        """Verify modified files need indexing."""
        vector_store._state["/existing/file.py"] = "old_hash"
        assert vector_store.needs_indexing("/existing/file.py", "new_hash") is True

    def test_needs_indexing_unchanged_file(self, vector_store):
        """Verify unchanged files don't need indexing."""
        vector_store._state["/existing/file.py"] = "same_hash"
        assert vector_store.needs_indexing("/existing/file.py", "same_hash") is False

    def test_known_files_returns_all_tracked(self, vector_store):
        """Verify known_files returns all tracked file paths."""
        vector_store._state = {
            "/path/to/file1.py": "hash1",
            "/path/to/file2.py": "hash2",
        }
        
        known = vector_store.known_files()
        
        assert len(known) == 2
        assert "/path/to/file1.py" in known
        assert "/path/to/file2.py" in known


class TestUpsertFile:
    """Tests for upsert_file operation."""

    @pytest.fixture
    def vector_store(self, tmp_path):
        """Create a VectorStore with mocked collection."""
        persist_dir = tmp_path / "chroma_test"
        persist_dir.mkdir()
        
        config = Config(
            chroma=ChromaConfig(persist_dir=str(persist_dir))
        )
        
        with patch('src.vectorstore.chromadb.PersistentClient') as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            
            mock_code_col = MagicMock()
            mock_session_col = MagicMock()
            mock_instance.get_or_create_collection.side_effect = [mock_code_col, mock_session_col]
            
            store = VectorStore(config)
            store.code_col = mock_code_col
        
        return store

    def test_upsert_empty_chunks_does_nothing(self, vector_store):
        """Verify empty chunk list results in no operation."""
        vector_store.upsert_file([], [], "hash123")
        
        vector_store.code_col.add.assert_not_called()
        vector_store.code_col.delete.assert_not_called()

    def test_upsert_deletes_old_version(self, vector_store, sample_python_code):
        """Verify old chunks are deleted before inserting new ones."""
        from src.chunker import Chunk
        
        chunks = [
            Chunk(
                content=sample_python_code[:100],
                file_path="/test/file.py",
                start_line=1,
                end_line=5,
                language="python"
            )
        ]
        embeddings = [[0.1] * 384]
        file_hash = "new_hash_456"
        
        # Mock get to return existing chunks
        vector_store.code_col.get.return_value = {"ids": ["old_id_1", "old_id_2"]}
        
        vector_store.upsert_file(chunks, embeddings, file_hash)
        
        vector_store.code_col.delete.assert_called_once_with(ids=["old_id_1", "old_id_2"])

    def test_upsert_inserts_new_chunks(self, vector_store, sample_python_code):
        """Verify new chunks are inserted with correct data."""
        from src.chunker import Chunk
        
        chunks = [
            Chunk(
                content="chunk 1 content",
                file_path="/test/file.py",
                start_line=1,
                end_line=5,
                language="python"
            ),
            Chunk(
                content="chunk 2 content",
                file_path="/test/file.py",
                start_line=6,
                end_line=10,
                language="python"
            )
        ]
        embeddings = [[0.1] * 384, [0.2] * 384]
        file_hash = "test_hash"
        
        vector_store.code_col.get.return_value = {"ids": []}
        
        vector_store.upsert_file(chunks, embeddings, file_hash)
        
        assert vector_store.code_col.add.call_count == 1
        call_args = vector_store.code_col.add.call_args
        
        ids = call_args[1]["ids"]
        assert len(ids) == 2
        assert ids[0].startswith("test_hash_")
        
        documents = call_args[1]["documents"]
        assert documents[0] == "chunk 1 content"
        assert documents[1] == "chunk 2 content"
        
        metadatas = call_args[1]["metadatas"]
        assert all(m[FILE_ID_KEY] == file_hash for m in metadatas)

    def test_upsert_updates_state(self, vector_store, sample_python_code, tmp_path):
        """Verify file state is updated after upsert."""
        from src.chunker import Chunk
        
        chunks = [
            Chunk(
                content=sample_python_code[:100],
                file_path="/test/file.py",
                start_line=1,
                end_line=5,
                language="python"
            )
        ]
        embeddings = [[0.1] * 384]
        file_hash = "updated_hash"
        
        vector_store.code_col.get.return_value = {"ids": []}
        
        vector_store.upsert_file(chunks, embeddings, file_hash)
        
        assert vector_store._state["/test/file.py"] == file_hash
        
        # Verify state was saved to disk
        state_file = tmp_path / "chroma_test" / "file_state.json"
        saved_state = json.loads(state_file.read_text())
        assert saved_state["/test/file.py"] == file_hash

    def test_upsert_batches_large_inputs(self, vector_store, sample_python_code):
        """Verify large chunk lists are batched correctly."""
        from src.chunker import Chunk
        
        # Create 150 chunks (more than batch size of 100)
        chunks = [
            Chunk(
                content=f"chunk {i}",
                file_path="/test/file.py",
                start_line=i,
                end_line=i+1,
                language="python"
            )
            for i in range(150)
        ]
        embeddings = [[float(i)] * 384 for i in range(150)]
        file_hash = "batch_test_hash"
        
        vector_store.code_col.get.return_value = {"ids": []}
        
        vector_store.upsert_file(chunks, embeddings, file_hash)
        
        # Should be called twice (100 + 50)
        assert vector_store.code_col.add.call_count == 2
        
        first_call_ids = vector_store.code_col.add.call_args_list[0][1]["ids"]
        second_call_ids = vector_store.code_col.add.call_args_list[1][1]["ids"]
        
        assert len(first_call_ids) == 100
        assert len(second_call_ids) == 50


class TestRemoveFile:
    """Tests for remove_file operation."""

    @pytest.fixture
    def vector_store(self, tmp_path):
        """Create a VectorStore with mocked collection."""
        persist_dir = tmp_path / "chroma_test"
        persist_dir.mkdir()
        
        config = Config(
            chroma=ChromaConfig(persist_dir=str(persist_dir))
        )
        
        with patch('src.vectorstore.chromadb.PersistentClient') as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            
            mock_code_col = MagicMock()
            mock_session_col = MagicMock()
            mock_instance.get_or_create_collection.side_effect = [mock_code_col, mock_session_col]
            
            store = VectorStore(config)
            store.code_col = mock_code_col
        
        return store

    def test_remove_deletes_chunks(self, vector_store):
        """Verify remove_file deletes associated chunks."""
        vector_store.code_col.get.return_value = {"ids": ["chunk_1", "chunk_2"]}
        
        vector_store.remove_file("/test/file.py")
        
        vector_store.code_col.get.assert_called_once_with(where={"file_path": "/test/file.py"})
        vector_store.code_col.delete.assert_called_once_with(ids=["chunk_1", "chunk_2"])

    def test_remove_updates_state(self, vector_store):
        """Verify remove_file removes entry from state."""
        vector_store._state = {
            "/test/file.py": "hash123",
            "/other/file.py": "hash456"
        }
        
        vector_store.code_col.get.return_value = {"ids": []}
        vector_store.remove_file("/test/file.py")
        
        assert "/test/file.py" not in vector_store._state
        assert "/other/file.py" in vector_store._state

    def test_remove_saves_state(self, vector_store, tmp_path):
        """Verify state is persisted after removal."""
        vector_store._state = {"/test/file.py": "hash123"}
        vector_store.code_col.get.return_value = {"ids": []}
        
        vector_store.remove_file("/test/file.py")
        
        state_file = tmp_path / "chroma_test" / "file_state.json"
        saved_state = json.loads(state_file.read_text())
        assert "/test/file.py" not in saved_state


class TestQueryCode:
    """Tests for query_code functionality."""

    @pytest.fixture
    def vector_store(self, tmp_path):
        """Create a VectorStore with mocked collection."""
        persist_dir = tmp_path / "chroma_test"
        persist_dir.mkdir()
        
        config = Config(
            chroma=ChromaConfig(persist_dir=str(persist_dir))
        )
        
        with patch('src.vectorstore.chromadb.PersistentClient') as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            
            mock_code_col = MagicMock()
            mock_session_col = MagicMock()
            mock_instance.get_or_create_collection.side_effect = [mock_code_col, mock_session_col]
            
            store = VectorStore(config)
            store.code_col = mock_code_col
        
        return store

    def test_query_returns_formatted_results(self, vector_store):
        """Verify query results are properly formatted."""
        embedding = [0.1] * 384
        
        vector_store.code_col.count.return_value = 10
        vector_store.code_col.query.return_value = {
            "documents": [["document content"]],
            "metadatas": [[{"file_path": "/test.py", "language": "python"}]],
            "distances": [[0.2]]
        }
        
        results = vector_store.query_code(embedding, n=5)
        
        assert len(results) == 1
        assert results[0]["content"] == "document content"
        assert results[0]["metadata"]["file_path"] == "/test.py"
        assert results[0]["score"] == 0.8  # 1.0 - 0.2

    def test_query_builds_language_filter(self, vector_store):
        """Verify language filter is applied correctly."""
        embedding = [0.1] * 384
        vector_store.code_col.count.return_value = 10
        vector_store.code_col.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        
        vector_store.query_code(embedding, language="python")
        
        where_clause = vector_store.code_col.query.call_args[1]["where"]
        assert where_clause == {"language": {"$eq": "python"}}

    def test_query_builds_path_filter(self, vector_store):
        """Verify path contains filter is applied correctly."""
        embedding = [0.1] * 384
        vector_store.code_col.count.return_value = 10
        vector_store.code_col.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        
        vector_store.query_code(embedding, path_contains="src/utils")
        
        where_clause = vector_store.code_col.query.call_args[1]["where"]
        assert where_clause == {"file_path": {"$contains": "src/utils"}}

    def test_query_combines_filters(self, vector_store):
        """Verify multiple filters are combined with $and."""
        embedding = [0.1] * 384
        vector_store.code_col.count.return_value = 10
        vector_store.code_col.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        
        vector_store.query_code(embedding, language="python", path_contains="utils")
        
        where_clause = vector_store.code_col.query.call_args[1]["where"]
        assert "$and" in where_clause
        assert {"language": {"$eq": "python"}} in where_clause["$and"]
        assert {"file_path": {"$contains": "utils"}} in where_clause["$and"]


class TestSessionCollection:
    """Tests for session collection operations."""

    @pytest.fixture
    def vector_store(self, tmp_path):
        """Create a VectorStore with mocked collection."""
        persist_dir = tmp_path / "chroma_test"
        persist_dir.mkdir()
        
        config = Config(
            chroma=ChromaConfig(persist_dir=str(persist_dir))
        )
        
        with patch('src.vectorstore.chromadb.PersistentClient') as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            
            mock_code_col = MagicMock()
            mock_session_col = MagicMock()
            mock_instance.get_or_create_collection.side_effect = [mock_code_col, mock_session_col]
            
            store = VectorStore(config)
            store.session_col = mock_session_col
        
        return store

    def test_add_session_entry(self, vector_store):
        """Verify session entries are added correctly."""
        entry_id = "session_123"
        document = "Q: What is this?\nA: This is a test."
        embedding = [0.5] * 384
        metadata = {"type": "conversation", "ts": "1234567890"}
        
        vector_store.add_session_entry(entry_id, document, embedding, metadata)
        
        vector_store.session_col.add.assert_called_once_with(
            ids=[entry_id],
            documents=[document],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    def test_trim_sessions_when_under_limit(self, vector_store):
        """Verify trim does nothing when under limit."""
        vector_store.session_col.count.return_value = 50
        
        vector_store.trim_sessions(max_count=100)
        
        vector_store.session_col.delete.assert_not_called()

    def test_trim_sessions_removes_oldest(self, vector_store):
        """Verify trim removes oldest entries when over limit."""
        vector_store.session_col.count.return_value = 120
        vector_store.session_col.get.return_value = {
            "ids": [f"old_id_{i}" for i in range(20)]
        }
        
        vector_store.trim_sessions(max_count=100)
        
        vector_store.session_col.delete.assert_called_once_with(
            ids=[f"old_id_{i}" for i in range(20)]
        )


class TestStats:
    """Tests for statistics reporting."""

    @pytest.fixture
    def vector_store(self, tmp_path):
        """Create a VectorStore with mocked collections."""
        persist_dir = tmp_path / "chroma_test"
        persist_dir.mkdir()
        
        config = Config(
            chroma=ChromaConfig(persist_dir=str(persist_dir))
        )
        
        with patch('src.vectorstore.chromadb.PersistentClient') as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            
            mock_code_col = MagicMock()
            mock_session_col = MagicMock()
            mock_instance.get_or_create_collection.side_effect = [mock_code_col, mock_session_col]
            
            store = VectorStore(config)
            store.code_col = mock_code_col
            store.session_col = mock_session_col
        
        return store

    def test_stats_returns_counts(self, vector_store):
        """Verify stats returns correct counts."""
        vector_store.code_col.count.return_value = 150
        vector_store.session_col.count.return_value = 45
        vector_store._state = {"/file1.py": "h1", "/file2.py": "h2", "/file3.py": "h3"}
        
        stats = vector_store.stats
        
        assert stats["code_chunks"] == 150
        assert stats["session_chunks"] == 45
        assert stats["indexed_files"] == 3
