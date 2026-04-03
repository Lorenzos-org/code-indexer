"""Unit tests for the discovery module."""
import pytest
import tempfile
import os
from pathlib import Path
from src.discovery import FileDiscovery


class TestFileDiscoveryInit:
    """Tests for FileDiscovery initialization."""

    def test_basic_initialization(self):
        """Test basic FileDiscovery creation."""
        fd = FileDiscovery(
            extensions={".py", ".js"},
            exclude_dirs={"__pycache__", "node_modules"},
            respect_gitignore=True,
            max_file_bytes=1024 * 500
        )
        assert fd.extensions == {".py", ".js"}
        assert fd.exclude_dirs == {"__pycache__", "node_modules"}
        assert fd.respect_gitignore is True
        assert fd.max_file_bytes == 1024 * 500


class TestFileDiscoveryDiscover:
    """Tests for FileDiscovery.discover method."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory structure for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some test files
            Path(tmpdir, "test.py").write_text("print('hello')")
            Path(tmpdir, "test.js").write_text("console.log('hello')")
            Path(tmpdir, "test.txt").write_text("hello")
            Path(tmpdir, "__pycache__").mkdir()
            Path(tmpdir, "__pycache__", "cached.pyc").write_text("cache")
            
            yield tmpdir

    def test_discover_python_files(self, temp_dir):
        """Test discovering only Python files."""
        fd = FileDiscovery(
            extensions={".py"},
            exclude_dirs=set(),
            respect_gitignore=False,
            max_file_bytes=1024 * 500
        )
        files = list(fd.discover(temp_dir))
        assert len(files) == 1
        assert files[0].name == "test.py"

    def test_discover_multiple_extensions(self, temp_dir):
        """Test discovering multiple file types."""
        fd = FileDiscovery(
            extensions={".py", ".js"},
            exclude_dirs=set(),
            respect_gitignore=False,
            max_file_bytes=1024 * 500
        )
        files = list(fd.discover(temp_dir))
        assert len(files) == 2
        extensions = {f.suffix for f in files}
        assert extensions == {".py", ".js"}

    def test_exclude_directories(self, temp_dir):
        """Test that excluded directories are skipped."""
        fd = FileDiscovery(
            extensions={".py", ".pyc"},
            exclude_dirs={"__pycache__"},
            respect_gitignore=False,
            max_file_bytes=1024 * 500
        )
        files = list(fd.discover(temp_dir))
        # Should find test.py but not cached.pyc
        assert len(files) == 1
        assert files[0].name == "test.py"

    def test_nonexistent_root(self):
        """Test discovering from non-existent directory."""
        fd = FileDiscovery(
            extensions={".py"},
            exclude_dirs=set(),
            respect_gitignore=False,
            max_file_bytes=1024 * 500
        )
        files = list(fd.discover("/nonexistent/path"))
        assert files == []

    def test_max_file_size_limit(self):
        """Test that files exceeding size limit are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file larger than 10 bytes
            large_file = Path(tmpdir, "large.py")
            large_file.write_text("x = 1\n" * 10)  # 60 bytes
            
            fd = FileDiscovery(
                extensions={".py"},
                exclude_dirs=set(),
                respect_gitignore=False,
                max_file_bytes=10  # 10 bytes limit
            )
            files = list(fd.discover(tmpdir))
            # File should be skipped due to size
            assert files == []


class TestGitignoreParsing:
    """Tests for gitignore parsing functionality."""

    def test_parse_gitignore(self):
        """Test parsing a .gitignore file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gitignore_path = Path(tmpdir, ".gitignore")
            gitignore_path.write_text("""# Comment
*.log
build/
secret.txt
""")
            
            patterns = FileDiscovery._parse_gitignore(gitignore_path)
            assert "*.log" in patterns
            assert "build/" in patterns
            assert "secret.txt" in patterns
            assert "# Comment" not in patterns

    def test_parse_nonexistent_gitignore(self):
        """Test parsing non-existent .gitignore returns empty list."""
        patterns = FileDiscovery._parse_gitignore(Path("/nonexistent/.gitignore"))
        assert patterns == []

    def test_gitignore_match_directory_pattern(self):
        """Test matching directory patterns in gitignore."""
        patterns = ["build/", "dist/"]
        rel = Path("build/output.txt")
        assert FileDiscovery._gitignore_match(rel, patterns) is True

    def test_gitignore_match_wildcard_pattern(self):
        """Test matching wildcard patterns in gitignore."""
        patterns = ["*.log", "*.tmp"]
        rel = Path("debug.log")
        assert FileDiscovery._gitignore_match(rel, patterns) is True
        
        rel2 = Path("subdir/error.log")
        assert FileDiscovery._gitignore_match(rel2, patterns) is True

    def test_gitignore_match_exact_pattern(self):
        """Test matching exact patterns in gitignore."""
        patterns = ["secret.txt", ".env"]
        rel = Path("secret.txt")
        assert FileDiscovery._gitignore_match(rel, patterns) is True
        
        rel2 = Path(".env")
        assert FileDiscovery._gitignore_match(rel2, patterns) is True

    def test_gitignore_no_match(self):
        """Test when no gitignore patterns match."""
        patterns = ["*.log", "build/"]
        rel = Path("src/main.py")
        assert FileDiscovery._gitignore_match(rel, patterns) is False


class TestFileDiscoveryWithGitignore:
    """Integration tests for FileDiscovery with gitignore."""

    def test_discover_respects_gitignore(self):
        """Test that discover respects .gitignore patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .gitignore
            gitignore = Path(tmpdir, ".gitignore")
            gitignore.write_text("*.log\n")
            
            # Create files
            Path(tmpdir, "main.py").write_text("print('hi')")
            Path(tmpdir, "debug.log").write_text("log data")
            
            fd = FileDiscovery(
                extensions={".py", ".log"},
                exclude_dirs=set(),
                respect_gitignore=True,
                max_file_bytes=1024 * 500
            )
            files = list(fd.discover(tmpdir))
            # Should only find main.py, not debug.log
            assert len(files) == 1
            assert files[0].name == "main.py"

    def test_discover_ignores_gitignore_when_disabled(self):
        """Test that discover ignores .gitignore when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .gitignore
            gitignore = Path(tmpdir, ".gitignore")
            gitignore.write_text("*.log\n")
            
            # Create files
            Path(tmpdir, "main.py").write_text("print('hi')")
            Path(tmpdir, "debug.log").write_text("log data")
            
            fd = FileDiscovery(
                extensions={".py", ".log"},
                exclude_dirs=set(),
                respect_gitignore=False,
                max_file_bytes=1024 * 500
            )
            files = list(fd.discover(tmpdir))
            # Should find both files
            assert len(files) == 2


class TestFileHash:
    """Tests for file_hash static method."""

    def test_file_hash_consistency(self):
        """Test that same content produces same hash."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name
        try:
            hash1 = FileDiscovery.file_hash(temp_path)
            hash2 = FileDiscovery.file_hash(temp_path)
            assert hash1 == hash2
        finally:
            Path(temp_path).unlink()

    def test_file_hash_different_content(self):
        """Test that different content produces different hash."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f1:
            f1.write("content 1")
            path1 = f1.name
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f2:
            f2.write("content 2")
            path2 = f2.name
        try:
            hash1 = FileDiscovery.file_hash(path1)
            hash2 = FileDiscovery.file_hash(path2)
            assert hash1 != hash2
        finally:
            Path(path1).unlink()
            Path(path2).unlink()

    def test_file_hash_nonexistent_file(self):
        """Test hashing non-existent file returns empty string."""
        result = FileDiscovery.file_hash("/nonexistent/file.txt")
        assert result == ""

    def test_file_hash_empty_file(self):
        """Test hashing an empty file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("")
            temp_path = f.name
        try:
            result = FileDiscovery.file_hash(temp_path)
            # Should return valid MD5 hash of empty content
            assert len(result) == 32
        finally:
            Path(temp_path).unlink()

    def test_file_hash_md5_format(self):
        """Test that hash is in MD5 format (32 hex chars)."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test")
            temp_path = f.name
        try:
            result = FileDiscovery.file_hash(temp_path)
            assert len(result) == 32
            assert all(c in '0123456789abcdef' for c in result)
        finally:
            Path(temp_path).unlink()
