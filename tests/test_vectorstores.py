"""Tests for vector store backends (structural tests)."""

from src.vectorstores.base import (
    Document,
    SearchResult,
    create_vector_store,
)
from src.vectorstores.lancedb_store import LanceDBStore
from src.vectorstores.pgvector_store import PgVectorStore

import pytest


def test_create_pgvector_store():
    """Factory creates PgVectorStore."""
    store = create_vector_store("pgvector", dsn="postgresql://test:test@localhost/test")
    assert isinstance(store, PgVectorStore)
    assert store.backend_name == "pgvector"


def test_create_lancedb_store():
    """Factory creates LanceDBStore."""
    store = create_vector_store("lancedb", path="/tmp/test_lance")
    assert isinstance(store, LanceDBStore)
    assert store.backend_name == "lancedb"


def test_create_unknown_store():
    """Factory raises for unknown backend."""
    with pytest.raises(ValueError, match="Unknown vector store"):
        create_vector_store("nonexistent")


def test_document_dataclass():
    """Document dataclass has expected fields."""
    doc = Document(
        content="Test content",
        persona="test_persona",
        source_file="test.vtt",
        video_title="Test Video",
    )
    assert doc.content == "Test content"
    assert doc.persona == "test_persona"
    assert doc.embedding == []


def test_search_result_dataclass():
    """SearchResult dataclass has expected fields."""
    doc = Document(content="Test", persona="test")
    result = SearchResult(document=doc, score=0.95, rank=0)
    assert result.score == 0.95
    assert result.rank == 0


@pytest.mark.asyncio
async def test_lancedb_lifecycle(tmp_path):
    """LanceDB store can initialize, add, count, search, and delete."""
    store = LanceDBStore(path=str(tmp_path / "lance_test"))
    await store.initialize()

    # Empty initially
    count = await store.count("nomic-embed-text")
    assert count == 0

    # Add documents with fake embeddings
    docs = [
        Document(
            content=f"Test document {i}",
            persona="test_persona",
            source_file="test.vtt",
            embedding=[float(i)] * 768,
        )
        for i in range(5)
    ]
    added = await store.add_documents(docs, "nomic-embed-text")
    assert added == 5

    # Count
    count = await store.count("nomic-embed-text")
    assert count == 5

    # Search
    results = await store.search(
        query_embedding=[1.0] * 768,
        embedding_model="nomic-embed-text",
        top_k=3,
    )
    assert len(results) == 3
    assert all(isinstance(r, SearchResult) for r in results)

    # Delete
    deleted = await store.delete_by_persona("test_persona", "nomic-embed-text")
    assert deleted == 5

    await store.close()
