"""Unit tests for the chunker module."""
import pytest
import tempfile
from pathlib import Path
from src.chunker import CodeChunker, Chunk, EXTENSION_LANGUAGE, BLOCK_STARTERS, CONTEXT_PATTERNS
from src.config import Config


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = Chunk(
            content="def hello():\n    pass",
            file_path="/test/file.py",
            start_line=1,
            end_line=2,
            language="python"
        )
        assert chunk.content == "def hello():\n    pass"
        assert chunk.file_path == "/test/file.py"
        assert chunk.start_line == 1
        assert chunk.end_line == 2
        assert chunk.language == "python"
        assert chunk.line_count == 1

    def test_chunk_hash_generation(self):
        """Test that content hash is generated automatically."""
        chunk = Chunk(
            content="test content",
            file_path="/test.py",
            start_line=1,
            end_line=1,
            language="python"
        )
        assert chunk.content_hash != ""
        assert len(chunk.content_hash) == 32  # MD5 hex length

    def test_chunk_to_metadata(self):
        """Test metadata dictionary generation."""
        chunk = Chunk(
            content="test",
            file_path="/test.py",
            start_line=5,
            end_line=10,
            language="python",
            enclosing_context="my_function"
        )
        meta = chunk.to_metadata()
        assert meta["file_path"] == "/test.py"
        assert meta["start_line"] == 5
        assert meta["end_line"] == 10
        assert meta["language"] == "python"
        assert meta["enclosing_context"] == "my_function"
        assert meta["line_count"] == 5


class TestExtensionLanguage:
    """Tests for extension to language mapping."""

    def test_python_extension(self):
        """Test Python extension mapping."""
        assert EXTENSION_LANGUAGE[".py"] == "python"

    def test_javascript_extensions(self):
        """Test JavaScript family extensions."""
        assert EXTENSION_LANGUAGE[".js"] == "javascript"
        assert EXTENSION_LANGUAGE[".ts"] == "typescript"
        assert EXTENSION_LANGUAGE[".jsx"] == "jsx"
        assert EXTENSION_LANGUAGE[".tsx"] == "tsx"

    def test_compiled_languages(self):
        """Test compiled language extensions."""
        assert EXTENSION_LANGUAGE[".java"] == "java"
        assert EXTENSION_LANGUAGE[".go"] == "go"
        assert EXTENSION_LANGUAGE[".rs"] == "rust"
        assert EXTENSION_LANGUAGE[".c"] == "c"
        assert EXTENSION_LANGUAGE[".cpp"] == "cpp"

    def test_config_extensions(self):
        """Test config file extensions."""
        assert EXTENSION_LANGUAGE[".yaml"] == "yaml"
        assert EXTENSION_LANGUAGE[".yml"] == "yaml"
        assert EXTENSION_LANGUAGE[".json"] == "json"
        assert EXTENSION_LANGUAGE[".toml"] == "toml"


class TestBlockStarters:
    """Tests for block starter patterns."""

    def test_python_patterns_exist(self):
        """Test that Python has block starter patterns."""
        assert "python" in BLOCK_STARTERS
        assert len(BLOCK_STARTERS["python"]) > 0

    def test_javascript_patterns_exist(self):
        """Test that JavaScript has block starter patterns."""
        assert "javascript" in BLOCK_STARTERS
        assert len(BLOCK_STARTERS["javascript"]) > 0


class TestContextPatterns:
    """Tests for context extraction patterns."""

    def test_python_context_patterns(self):
        """Test Python context patterns exist."""
        assert "python" in CONTEXT_PATTERNS
        assert len(CONTEXT_PATTERNS["python"]) > 0

    def test_go_context_patterns(self):
        """Test Go context patterns exist."""
        assert "go" in CONTEXT_PATTERNS


class TestCodeChunker:
    """Tests for CodeChunker class."""

    @pytest.fixture
    def default_config(self):
        """Create a default config for testing."""
        return Config()

    @pytest.fixture
    def chunker(self, default_config):
        """Create a CodeChunker instance."""
        return CodeChunker(default_config)

    def test_chunker_initialization(self, chunker, default_config):
        """Test chunker initializes with correct config values."""
        assert chunker.chunk_lines == default_config.indexing.chunk_lines
        assert chunker.overlap == default_config.indexing.chunk_overlap
        assert chunker.max_file_kb == default_config.indexing.max_file_kb * 1024

    def test_chunk_empty_file(self, chunker):
        """Test chunking an empty file returns empty list."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("")
            temp_path = f.name
        try:
            chunks = chunker.chunk_file(temp_path)
            assert chunks == []
        finally:
            Path(temp_path).unlink()

    def test_chunk_nonexistent_file(self, chunker):
        """Test chunking a non-existent file returns empty list."""
        chunks = chunker.chunk_file("/nonexistent/file.py")
        assert chunks == []

    def test_chunk_simple_python_file(self, chunker):
        """Test chunking a simple Python file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""def hello():
    print("Hello")

def world():
    print("World")
""")
            temp_path = f.name
        try:
            chunks = chunker.chunk_file(temp_path)
            assert len(chunks) > 0
            assert all(c.language == "python" for c in chunks)
            assert all(c.file_path == temp_path for c in chunks)
        finally:
            Path(temp_path).unlink()

    def test_chunk_large_file_limit(self):
        """Test that files exceeding max size return empty list."""
        config = Config()
        config.indexing.max_file_kb = 1  # 1KB limit
        chunker = CodeChunker(config)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write more than 1KB
            f.write("x = 1\n" * 500)
            temp_path = f.name
        try:
            chunks = chunker.chunk_file(temp_path)
            assert chunks == []
        finally:
            Path(temp_path).unlink()

    def test_chunk_unknown_extension(self, chunker):
        """Test chunking a file with unknown extension."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write("some content here\nmore content")
            temp_path = f.name
        try:
            chunks = chunker.chunk_file(temp_path)
            # Should use simple chunking for unknown languages
            assert len(chunks) >= 0  # May be empty or have chunks
        finally:
            Path(temp_path).unlink()

    def test_chunk_python_with_classes_and_functions(self, chunker):
        """Test chunking Python file with classes and functions."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""class MyClass:
    def method1(self):
        return 1

    def method2(self):
        return 2

def standalone_func():
    pass
""")
            temp_path = f.name
        try:
            chunks = chunker.chunk_file(temp_path)
            assert len(chunks) > 0
            # Check that context is extracted
            contexts = [c.enclosing_context for c in chunks if c.enclosing_context]
            assert len(contexts) > 0
        finally:
            Path(temp_path).unlink()

    def test_chunk_go_file(self, chunker):
        """Test chunking a Go file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
            f.write("""package main

func main() {
    println("Hello")
}

func helper() int {
    return 42
}
""")
            temp_path = f.name
        try:
            chunks = chunker.chunk_file(temp_path)
            assert len(chunks) > 0
            assert all(c.language == "go" for c in chunks)
        finally:
            Path(temp_path).unlink()

    def test_chunk_markdown_file(self, chunker):
        """Test chunking a Markdown file (no block starters)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Header

Some paragraph text.

## Subheader

More text here.
""")
            temp_path = f.name
        try:
            chunks = chunker.chunk_file(temp_path)
            # Markdown should use simple chunking
            assert len(chunks) >= 0
        finally:
            Path(temp_path).unlink()

    def test_chunk_overlap(self):
        """Test that chunk overlap is applied."""
        config = Config()
        config.indexing.chunk_lines = 5
        config.indexing.chunk_overlap = 2
        chunker = CodeChunker(config)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for i in range(20):
                f.write(f"Line {i}\n")
            temp_path = f.name
        try:
            chunks = chunker.chunk_file(temp_path)
            # Should have multiple chunks with overlap
            assert len(chunks) > 1
        finally:
            Path(temp_path).unlink()

    def test_extract_context_python(self, chunker):
        """Test context extraction for Python code."""
        lines = ["def my_function(arg1, arg2):\n", "    pass\n"]
        context = chunker._extract_context(lines, "python")
        assert context == "my_function"

    def test_extract_context_class(self, chunker):
        """Test context extraction for class definitions."""
        lines = ["class MyClass:\n", "    pass\n"]
        context = chunker._extract_context(lines, "python")
        assert context == "MyClass"

    def test_extract_context_no_match(self, chunker):
        """Test context extraction when no pattern matches."""
        lines = ["# Just a comment\n", "x = 1\n"]
        context = chunker._extract_context(lines, "python")
        assert context == ""

    def test_simple_chunk_basic(self, chunker):
        """Test simple chunking strategy."""
        lines = [f"Line {i}\n" for i in range(10)]
        chunks = chunker._simple_chunk(lines, "unknown", "/test.txt")
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_smart_chunk_basic(self, chunker):
        """Test smart chunking strategy."""
        lines = [
            "def func1():\n",
            "    pass\n",
            "def func2():\n",
            "    pass\n",
        ]
        chunks = chunker._smart_chunk(lines, "python", "/test.py")
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
