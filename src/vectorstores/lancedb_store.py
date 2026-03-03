"""LanceDB embedded vector store backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from .base import Document, SearchResult, VectorStore

logger = logging.getLogger(__name__)


def _table_name(embedding_model: str) -> str:
    """Generate table name from embedding model."""
    return f"documents_{embedding_model.replace('-', '_')}"


class LanceDBStore(VectorStore):
    """LanceDB embedded vector store.

    Runs in-process (no server). Data stored as Lance columnar files.
    """

    def __init__(self, path: str = "./data/vectordb/lancedb", **kwargs: Any) -> None:
        self._path = Path(path)
        self._db: lancedb.DBConnection | None = None

    @property
    def backend_name(self) -> str:
        return "lancedb"

    def _get_db(self) -> lancedb.DBConnection:
        if self._db is None:
            self._path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self._path))
        return self._db

    async def initialize(self) -> None:
        """Ensure the database directory exists."""
        self._get_db()
        logger.info("LanceDB store initialized at %s", self._path)

    async def add_documents(
        self, documents: list[Document], embedding_model: str
    ) -> int:
        """Add documents to a LanceDB table."""
        if not documents:
            return 0

        db = self._get_db()
        table_name = _table_name(embedding_model)

        # Build records
        records = []
        for doc in documents:
            if not doc.embedding:
                logger.warning("Skipping document without embedding")
                continue
            records.append({
                "content": doc.content,
                "persona": doc.persona,
                "source_file": doc.source_file,
                "video_title": doc.video_title,
                "video_date": doc.video_date,
                "video_url": doc.video_url,
                "chunk_index": doc.chunk_index,
                "vector": doc.embedding,
            })

        if not records:
            return 0

        # Create or append to table
        if table_name in db.table_names():
            table = db.open_table(table_name)
            table.add(records)
        else:
            db.create_table(table_name, records)

        logger.info(
            "Added %d documents to LanceDB table '%s'",
            len(records), table_name,
        )
        return len(records)

    async def search(
        self,
        query_embedding: list[float],
        embedding_model: str,
        persona: str | None = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Vector similarity search in LanceDB."""
        db = self._get_db()
        table_name = _table_name(embedding_model)

        if table_name not in db.table_names():
            logger.warning("Table '%s' does not exist", table_name)
            return []

        table = db.open_table(table_name)
        query = table.search(query_embedding).limit(top_k)

        if persona:
            query = query.where(f"persona = '{persona}'")

        raw_results = query.to_list()

        results: list[SearchResult] = []
        for rank, row in enumerate(raw_results):
            doc = Document(
                content=row["content"],
                persona=row["persona"],
                source_file=row.get("source_file", ""),
                video_title=row.get("video_title", ""),
                video_date=row.get("video_date", ""),
                video_url=row.get("video_url", ""),
                chunk_index=row.get("chunk_index", 0),
            )
            # LanceDB returns _distance (L2) — convert to similarity
            distance = row.get("_distance", 0.0)
            similarity = 1.0 / (1.0 + distance)
            results.append(
                SearchResult(document=doc, score=similarity, rank=rank)
            )

        return results

    async def count(
        self, embedding_model: str, persona: str | None = None
    ) -> int:
        db = self._get_db()
        table_name = _table_name(embedding_model)

        if table_name not in db.table_names():
            return 0

        table = db.open_table(table_name)

        if persona:
            return len(
                table.search().where(f"persona = '{persona}'").to_list()
            )
        return table.count_rows()

    async def delete_by_persona(
        self, persona: str, embedding_model: str
    ) -> int:
        db = self._get_db()
        table_name = _table_name(embedding_model)

        if table_name not in db.table_names():
            return 0

        table = db.open_table(table_name)
        before = table.count_rows()
        table.delete(f"persona = '{persona}'")
        after = table.count_rows()
        return before - after

    async def close(self) -> None:
        self._db = None
        logger.info("LanceDB store closed")
