"""Unit tests for the embedder module."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.embedder import OllamaEmbedder
from src.config import Config


class TestOllamaEmbedderInit:
    """Tests for OllamaEmbedder initialization."""

    def test_default_config(self):
        """Test embedder initializes with config values."""
        config = Config()
        embedder = OllamaEmbedder(config)
        
        assert embedder.base_url == config.ollama.base_url
        assert embedder.model == config.ollama.model
        assert embedder.timeout == config.ollama.timeout
        assert embedder.batch_size == config.ollama.batch_size
        assert embedder.num_threads == config.ollama.num_threads
        assert embedder.delay == config.ollama.inter_batch_delay

    def test_custom_config(self):
        """Test embedder with custom config values."""
        config = Config()
        config.ollama.base_url = "http://custom:11434"
        config.ollama.model = "custom-model"
        config.ollama.timeout = 60
        config.ollama.batch_size = 16
        
        embedder = OllamaEmbedder(config)
        
        assert embedder.base_url == "http://custom:11434"
        assert embedder.model == "custom-model"
        assert embedder.timeout == 60
        assert embedder.batch_size == 16


class TestKeyGeneration:
    """Tests for cache key generation."""

    def test_key_generation(self):
        """Test that keys are generated consistently."""
        config = Config()
        embedder = OllamaEmbedder(config)
        
        key1 = embedder._key("test text")
        key2 = embedder._key("test text")
        key3 = embedder._key("different text")
        
        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 20  # SHA256 hex truncated to 20 chars


class TestEmbedderCache:
    """Tests for embedder caching functionality."""

    @pytest.fixture
    def embedder(self):
        """Create an embedder instance."""
        config = Config()
        return OllamaEmbedder(config)

    def test_initial_cache_empty(self, embedder):
        """Test that cache starts empty."""
        assert len(embedder._cache) == 0

    def test_clear_cache(self, embedder):
        """Test clearing the cache."""
        embedder._cache["key1"] = [0.1, 0.2]
        embedder._cache["key2"] = [0.3, 0.4]
        
        embedder.clear_cache()
        
        assert len(embedder._cache) == 0

    def test_stats_initial(self, embedder):
        """Test initial stats values."""
        stats = embedder.stats
        assert stats["cache_size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["errors"] == 0


class TestEmbedMethod:
    """Tests for the embed method."""

    @patch('src.embedder.requests.post')
    def test_embed_success(self, mock_post):
        """Test successful embedding."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        config = Config()
        embedder = OllamaEmbedder(config)
        
        result = embedder.embed("test text")
        
        assert result == [0.1, 0.2, 0.3]
        mock_post.assert_called_once()

    @patch('src.embedder.requests.post')
    def test_embed_uses_cache(self, mock_post):
        """Test that cached embeddings are returned without API call."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_post.return_value = mock_response
        
        config = Config()
        embedder = OllamaEmbedder(config)
        
        # First call - should hit API
        result1 = embedder.embed("test text")
        assert mock_post.call_count == 1
        
        # Second call with same text - should use cache
        result2 = embedder.embed("test text")
        assert mock_post.call_count == 1  # No additional calls
        assert result1 == result2

    @patch('src.embedder.requests.post')
    def test_embed_error_handling(self, mock_post):
        """Test error handling during embedding."""
        mock_post.side_effect = Exception("API Error")
        
        config = Config()
        embedder = OllamaEmbedder(config)
        
        with pytest.raises(Exception):
            embedder.embed("test text")
        
        assert embedder.stats["errors"] == 1


class TestEmbedBatchMethod:
    """Tests for the embed_batch method."""

    @patch('src.embedder.requests.post')
    def test_embed_batch_success(self, mock_post):
        """Test successful batch embedding."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6]
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        config = Config()
        config.ollama.batch_size = 10
        embedder = OllamaEmbedder(config)
        
        texts = ["text1", "text2", "text3"]
        results = embedder.embed_batch(texts)
        
        assert len(results) == 3
        assert results[0] == [0.1, 0.2]
        assert results[1] == [0.3, 0.4]
        assert results[2] == [0.5, 0.6]

    @patch('src.embedder.requests.post')
    def test_embed_batch_with_cache(self, mock_post):
        """Test batch embedding with some cached items."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[0.5, 0.6]]
        }
        mock_post.return_value = mock_response
        
        config = Config()
        config.ollama.batch_size = 10
        embedder = OllamaEmbedder(config)
        
        # Pre-populate cache
        key = embedder._key("cached text")
        embedder._cache[key] = [0.1, 0.2]
        
        texts = ["cached text", "new text"]
        results = embedder.embed_batch(texts)
        
        assert len(results) == 2
        assert results[0] == [0.1, 0.2]  # From cache
        assert results[1] == [0.5, 0.6]  # From API
        assert embedder.stats["hits"] >= 1
        assert embedder.stats["misses"] >= 1

    @patch('src.embedder.requests.post')
    def test_embed_batch_multiple_batches(self, mock_post):
        """Test batch embedding with multiple batches."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2]]
        }
        mock_post.return_value = mock_response
        
        config = Config()
        config.ollama.batch_size = 2
        embedder = OllamaEmbedder(config)
        
        texts = ["text1", "text2", "text3", "text4", "text5"]
        results = embedder.embed_batch(texts)
        
        assert len(results) == 5
        # Should make 3 API calls (2+2+1)
        assert mock_post.call_count == 3

    @patch('src.embedder.requests.post')
    def test_embed_batch_error_handling(self, mock_post):
        """Test error handling in batch embedding."""
        mock_post.side_effect = Exception("Batch API Error")
        
        config = Config()
        config.ollama.batch_size = 10
        embedder = OllamaEmbedder(config)
        
        texts = ["text1", "text2"]
        with pytest.raises(Exception):
            embedder.embed_batch(texts)
        
        assert embedder.stats["errors"] == 1


class TestEmbedderStats:
    """Tests for embedder statistics tracking."""

    @patch('src.embedder.requests.post')
    def test_stats_tracking(self, mock_post):
        """Test that stats are tracked correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2]]}
        mock_post.return_value = mock_response
        
        config = Config()
        embedder = OllamaEmbedder(config)
        
        # Embed new text (miss)
        embedder.embed("text1")
        assert embedder.stats["misses"] == 1
        
        # Embed same text again (hit)
        embedder.embed("text1")
        assert embedder.stats["hits"] == 1
        assert embedder.stats["misses"] == 1
        
        # Check cache size
        assert embedder.stats["cache_size"] == 1

    def test_stats_cache_size(self):
        """Test cache size tracking."""
        config = Config()
        embedder = OllamaEmbedder(config)
        
        embedder._cache["key1"] = [0.1]
        embedder._cache["key2"] = [0.2]
        embedder._cache["key3"] = [0.3]
        
        assert embedder.stats["cache_size"] == 3
