"""
Centralized fixtures and mock data for the test suite.

This module provides reusable test data, mock objects, and pytest fixtures
to ensure consistency and reduce duplication across all test files.
"""
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from src.config import Config, OllamaConfig, ChromaConfig, IndexingConfig
from src.chunker import Chunk, CodeChunker


# =============================================================================
# 1. REALISTIC CODE SNIPPETS (Ground Truth for Chunking Tests)
# =============================================================================

PYTHON_CODE_SAMPLE = """
import os
import sys
from typing import List, Optional

class DataProcessor:
    \"\"\"Handles data processing logic.\"\"\"
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.cache = {}

    def _load_config(self, path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")
        # Simulate loading
        return {"setting": "value"}

    def process(self, items: List[str]) -> List[str]:
        results = []
        for item in items:
            if item in self.cache:
                results.append(self.cache[item])
            else:
                processed = self._transform(item)
                self.cache[item] = processed
                results.append(processed)
        return results

    def _transform(self, item: str) -> str:
        return item.upper()

def main():
    processor = DataProcessor("config.json")
    data = ["hello", "world"]
    print(processor.process(data))

if __name__ == "__main__":
    main()
"""

JAVASCRIPT_CODE_SAMPLE = """
import React, { useState, useEffect } from 'react';

interface UserProps {
  id: number;
  name: string;
}

const UserProfile: React.FC<UserProps> = ({ id, name }) => {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const response = await fetch(`/api/users/${id}`);
        const result = await response.json();
        setData(result);
      } catch (error) {
        console.error("Failed to fetch:", error);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, [id]);

  if (loading) return <div>Loading...</div>;

  return (
    <div className="profile">
      <h1>{name}</h1>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
};

export default UserProfile;
"""

RUST_CODE_SAMPLE = """
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub port: u16,
    pub host: String,
}

impl Config {
    pub fn new(port: u16, host: &str) -> Self {
        Config {
            port,
            host: host.to_string(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.port < 1024 {
            return Err("Port must be >= 1024".to_string());
        }
        Ok(())
    }
}

fn main() {
    let config = Config::new(8080, "localhost");
    match config.validate() {
        Ok(_) => println!("Server starting on {}:{}", config.host, config.port),
        Err(e) => eprintln!("Error: {}", e),
    }
}
"""


# =============================================================================
# 2. MOCK OBJECTS & HELPERS
# =============================================================================

def create_mock_embedding(dimensions: int = 384) -> List[float]:
    """Generates a deterministic pseudo-random embedding vector."""
    # Use a fixed seed logic for reproducibility if needed, 
    # but simple sin/cos wave is often enough for shape testing
    return [float(np.sin(i) * np.cos(i)) for i in range(dimensions)]


class MockOllamaClient:
    """Simulates Ollama API responses without network calls."""
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.call_count = 0
    
    def embeddings(self, model: str, prompt: str) -> Dict[str, Any]:
        self.call_count += 1
        # Simulate latency or specific errors if needed
        return {
            "embeddings": [create_mock_embedding()]
        }


class MockChromaCollection:
    """Simulates a ChromaDB Collection."""
    
    def __init__(self, name: str):
        self.name = name
        self.storage: Dict[str, Dict] = {}  # id -> {embedding, document, metadata}
    
    def add(self, ids: List[str], embeddings: List[List[float]], 
            documents: List[str], metadatas: List[Dict]):
        for i, doc_id in enumerate(ids):
            self.storage[doc_id] = {
                "embedding": embeddings[i],
                "document": documents[i],
                "metadata": metadatas[i] if metadatas else {}
            }
    
    def get(self, ids: List[str] = None, where: Dict = None) -> Dict:
        if ids is None:
            ids = list(self.storage.keys())
        
        result_ids = []
        result_docs = []
        result_metas = []
        result_embs = []
        
        for doc_id in ids:
            if doc_id in self.storage:
                item = self.storage[doc_id]
                # Simple 'where' filter simulation
                if where:
                    match = True
                    for k, v in where.items():
                        if item["metadata"].get(k) != v:
                            match = False
                            break
                    if not match:
                        continue
                
                result_ids.append(doc_id)
                result_docs.append(item["document"])
                result_metas.append(item["metadata"])
                result_embs.append(item["embedding"])
        
        return {
            "ids": result_ids,
            "documents": result_docs,
            "metadatas": result_metas,
            "embeddings": result_embs
        }
    
    def delete(self, ids: List[str]):
        for doc_id in ids:
            self.storage.pop(doc_id, None)
    
    def count(self) -> int:
        return len(self.storage)


# =============================================================================
# 3. PYTEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_python_code() -> str:
    return PYTHON_CODE_SAMPLE


@pytest.fixture
def sample_javascript_code() -> str:
    return JAVASCRIPT_CODE_SAMPLE


@pytest.fixture
def sample_rust_code() -> str:
    return RUST_CODE_SAMPLE


@pytest.fixture
def mock_ollama_client():
    """Returns a mock Ollama client instance."""
    return MockOllamaClient()


@pytest.fixture
def mock_chroma_collection():
    """Returns a mock ChromaDB collection instance."""
    return MockChromaCollection("test-collection")


@pytest.fixture
def temp_project_dir():
    """
    Creates a temporary directory with a realistic project structure.
    
    Structure:
    tmp_dir/
    ├── src/
    │   ├── main.py
    │   └── utils.py
    ├── tests/
    │   └── test_main.py
    ├── .gitignore
    └── README.md
    """
    root = tempfile.mkdtemp()
    root_path = Path(root)
    
    # Create directories
    (root_path / "src").mkdir()
    (root_path / "tests").mkdir()
    (root_path / "docs").mkdir()
    
    # Create files with content
    (root_path / "src" / "main.py").write_text(PYTHON_CODE_SAMPLE)
    (root_path / "src" / "utils.py").write_text("# Utility functions\n\ndef helper():\n    pass")
    (root_path / "tests" / "test_main.py").write_text("def test_main():\n    assert True")
    (root_path / "README.md").write_text("# Project\n\nDescription here.")
    
    # Create a .gitignore
    gitignore_content = """
__pycache__/
*.pyc
.env
venv/
.DS_Store
"""
    (root_path / ".gitignore").write_text(gitignore_content)
    
    yield root_path
    
    # Cleanup
    shutil.rmtree(root)


@pytest.fixture
def base_config():
    """Returns a valid base configuration object."""
    return Config(
        ollama=OllamaConfig(model="nomic-embed-text", host="http://localhost:11434"),
        chroma=ChromaConfig(path="./chroma_db", collection_name="test"),
        indexing=IndexingConfig(chunk_size=512, chunk_overlap=50)
    )


@pytest.fixture
def chunker_instance():
    """Returns a standard CodeChunker instance."""
    return CodeChunker(chunk_size=512, chunk_overlap=50)
