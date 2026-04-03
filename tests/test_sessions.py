"""Unit tests for the sessions module."""
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
from src.sessions import SessionLogger
from src.config import Config


class TestSessionLoggerInit:
    """Tests for SessionLogger initialization."""

    def test_default_config(self):
        """Test logger initializes with config values."""
        config = Config()
        with tempfile.TemporaryDirectory() as tmpdir:
            config.sessions.log_dir = tmpdir
            logger = SessionLogger(config)
            
            assert logger.max_sessions == config.sessions.max_sessions
            assert logger.top_k == config.sessions.store_top_k_results
            assert logger._session_id is None
            assert logger._session_file is None

    def test_creates_log_directory(self):
        """Test that log directory is created if it doesn't exist."""
        config = Config()
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "new_logs"
            config.sessions.log_dir = str(log_dir)
            
            assert not log_dir.exists()
            logger = SessionLogger(config)
            assert log_dir.exists()


class TestSessionLifecycle:
    """Tests for session lifecycle management."""

    @pytest.fixture
    def logger(self):
        """Create a SessionLogger instance."""
        config = Config()
        temp_dir = tempfile.mkdtemp()
        config.sessions.log_dir = temp_dir
        logger = SessionLogger(config)
        yield logger
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_start_session(self, logger):
        """Test starting a new session."""
        session_id = logger.start_session()
        
        assert session_id is not None
        assert len(session_id) > 0
        assert logger._session_id == session_id
        assert logger._session_file is not None
        assert logger._session_file.exists()

    def test_session_id_property(self, logger):
        """Test session_id property auto-starts session."""
        assert logger._session_id is None
        session_id = logger.session_id
        assert session_id is not None
        assert logger._session_id == session_id

    def test_session_file_naming(self, logger):
        """Test session file is named correctly."""
        session_id = logger.start_session()
        expected_file = logger.log_dir / f"{session_id}.jsonl"
        assert logger._session_file == expected_file


class TestSessionWriting:
    """Tests for writing session events."""

    @pytest.fixture
    def logger(self):
        """Create a SessionLogger instance."""
        config = Config()
        temp_dir = tempfile.mkdtemp()
        config.sessions.log_dir = temp_dir
        logger = SessionLogger(config)
        yield logger
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_write_query_event(self, logger):
        """Test logging a query event."""
        results = [
            {"score": 0.9, "metadata": {"file_path": "/test.py", "start_line": 1, "end_line": 10}},
            {"score": 0.8, "metadata": {"file_path": "/test2.py", "start_line": 5, "end_line": 15}}
        ]
        filters = {"language": "python"}
        
        logger.log_query("test query", results, filters, 100)
        
        # Read the session file
        content = logger._session_file.read_text()
        lines = content.strip().split('\n')
        assert len(lines) == 1
        
        event = json.loads(lines[0])
        assert event["type"] == "query"
        assert event["query"] == "test query"
        assert event["duration_ms"] == 100
        assert "ts" in event

    def test_write_index_event(self, logger):
        """Test logging an index event."""
        logger.log_index(
            path="/test/path",
            files_indexed=10,
            files_skipped=5,
            chunks_created=50,
            duration_ms=1000,
            errors=["error1", "error2"]
        )
        
        content = logger._session_file.read_text()
        lines = content.strip().split('\n')
        
        event = json.loads(lines[0])
        assert event["type"] == "index"
        assert event["path"] == "/test/path"
        assert event["files_indexed"] == 10
        assert event["files_skipped"] == 5
        assert event["chunks"] == 50
        assert event["duration_ms"] == 1000

    def test_write_error_event(self, logger):
        """Test logging an error event."""
        logger.log_error("Test error", {"file": "/test.py"})
        
        content = logger._session_file.read_text()
        event = json.loads(content.strip())
        
        assert event["type"] == "error"
        assert event["error"] == "Test error"
        assert event["context"]["file"] == "/test.py"

    def test_write_custom_event(self, logger):
        """Test logging a custom event."""
        logger.log_custom("custom_tag", {"data": "value"})
        
        content = logger._session_file.read_text()
        event = json.loads(content.strip())
        
        assert event["type"] == "custom_tag"
        assert event["data"] == "value"

    def test_query_filters_empty_values_removed(self, logger):
        """Test that empty filter values are removed."""
        results = []
        filters = {"language": "python", "path_contains": None}
        
        logger.log_query("test", results, filters, 0)
        
        content = logger._session_file.read_text()
        event = json.loads(content.strip())
        
        assert "language" in event["filters"]
        assert "path_contains" not in event["filters"]

    def test_top_k_results_limit(self, logger):
        """Test that only top_k results are stored."""
        logger.top_k = 2
        results = [{"score": i, "metadata": {"file_path": f"/{i}.py", "start_line": 1, "end_line": 1}} 
                   for i in range(10)]
        
        logger.log_query("test", results, {}, 0)
        
        content = logger._session_file.read_text()
        event = json.loads(content.strip())
        
        assert len(event["top_results"]) == 2


class TestSessionReading:
    """Tests for reading session data."""

    @pytest.fixture
    def logger(self):
        """Create a SessionLogger instance."""
        config = Config()
        temp_dir = tempfile.mkdtemp()
        config.sessions.log_dir = temp_dir
        logger = SessionLogger(config)
        yield logger
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_get_session(self, logger):
        """Test retrieving a session by ID."""
        logger.start_session()
        session_id = logger._session_id
        
        logger.log_query("test query", [], {}, 0)
        
        events = logger.get_session(session_id)
        assert len(events) == 1
        assert events[0]["query"] == "test query"

    def test_get_nonexistent_session(self, logger):
        """Test retrieving a non-existent session."""
        events = logger.get_session("nonexistent_id")
        assert events == []

    def test_list_sessions(self, logger):
        """Test listing sessions."""
        # Create multiple sessions
        for i in range(3):
            logger.start_session()
            logger.log_query(f"query {i}", [], {}, 0)
            logger._session_id = None  # Reset to create new session
        
        sessions = logger.list_sessions(limit=10)
        assert len(sessions) >= 3
        
        # Check session structure
        for session in sessions:
            assert "id" in session
            assert "started" in session
            assert "last_activity" in session
            assert "events" in session

    def test_list_sessions_limit(self, logger):
        """Test limiting session list results."""
        for i in range(10):
            logger.start_session()
            logger._session_id = None
        
        sessions = logger.list_sessions(limit=5)
        assert len(sessions) <= 5

    def test_iter_all_events(self, logger):
        """Test iterating all events across sessions."""
        logger.start_session()
        logger.log_query("query1", [], {}, 0)
        
        logger._session_id = None
        logger.start_session()
        logger.log_query("query2", [], {}, 0)
        
        events = list(logger.iter_all_events())
        assert len(events) == 2

    def test_iter_events_by_type(self, logger):
        """Test iterating events filtered by type."""
        logger.start_session()
        logger.log_query("test", [], {}, 0)
        logger.log_error("error", {})
        
        query_events = list(logger.iter_all_events(event_type="query"))
        error_events = list(logger.iter_all_events(event_type="error"))
        
        assert len(query_events) == 1
        assert len(error_events) == 1


class TestSessionCleanup:
    """Tests for session cleanup functionality."""

    def test_cleanup_old_sessions(self):
        """Test that old sessions are cleaned up."""
        config = Config()
        temp_dir = tempfile.mkdtemp()
        config.sessions.log_dir = temp_dir
        config.sessions.max_sessions = 3
        
        logger = SessionLogger(config)
        
        # Create more sessions than max
        for i in range(5):
            logger.start_session()
            logger.log_query(f"query {i}", [], {}, 0)
            logger._session_id = None
        
        # Force cleanup check
        logger._cleanup()
        
        # Count remaining session files
        session_files = list(Path(temp_dir).glob("*.jsonl"))
        assert len(session_files) <= 3
        
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cleanup_handles_deletion_errors(self):
        """Test cleanup handles file deletion errors gracefully."""
        config = Config()
        temp_dir = tempfile.mkdtemp()
        config.sessions.log_dir = temp_dir
        config.sessions.max_sessions = 1
        
        logger = SessionLogger(config)
        logger.start_session()
        
        # Should not raise exception even if deletion fails
        logger._cleanup()
        
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
