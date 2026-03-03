"""PostgreSQL + pgvector vector store backend."""

from __future__ import annotations

import logging
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from .base import Document, SearchResult, VectorStore

logger = logging.getLogger(__name__)

# Table name per embedding model
_TABLE_MAP = {
    "nomic-embed-text": "documents_nomic",
    "mxbai-embed-large": "documents_mxbai",
}


class PgVectorStore(VectorStore):
    """PostgreSQL + pgvector implementation.

    Uses separate tables per embedding model (different dimensions).
    Cosine similarity search via pgvector operators.
    """

    def __init__(self, dsn: str, **kwargs: Any) -> None:
        self._dsn = dsn
        self._conn: psycopg.AsyncConnection | None = None

    @property
    def backend_name(self) -> str:
        return "pgvector"

    def _table(self, embedding_model: str) -> str:
        if embedding_model not in _TABLE_MAP:
            raise ValueError(f"No table for model '{embedding_model}'")
        return _TABLE_MAP[embedding_model]

    async def _get_conn(self) -> psycopg.AsyncConnection:
        if self._conn is None or self._conn.closed:
            self._conn = await psycopg.AsyncConnection.connect(
                self._dsn, autocommit=True
            )
            await register_vector(self._conn)
        return self._conn

    async def initialize(self) -> None:
        """Verify connection and pgvector extension."""
        conn = await self._get_conn()
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
            )
            row = await cur.fetchone()
            if not row:
                raise RuntimeError("pgvector extension not installed")
        logger.info("pgvector store initialized")

    async def add_documents(
        self, documents: list[Document], embedding_model: str
    ) -> int:
        """Insert documents with embeddings into the appropriate table."""
        if not documents:
            return 0

        table = self._table(embedding_model)
        conn = await self._get_conn()

        inserted = 0
        async with conn.cursor() as cur:
            for doc in documents:
                if not doc.embedding:
                    logger.warning("Skipping document without embedding")
                    continue
                await cur.execute(
                    f"""
                    INSERT INTO {table}
                        (content, persona, source_file, video_title,
                         video_date, video_url, chunk_index, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        doc.content,
                        doc.persona,
                        doc.source_file,
                        doc.video_title,
                        doc.video_date or None,
                        doc.video_url,
                        doc.chunk_index,
                        doc.embedding,
                    ),
                )
                inserted += 1

        logger.info(
            "Inserted %d documents into %s", inserted, table
        )
        return inserted

    async def search(
        self,
        query_embedding: list[float],
        embedding_model: str,
        persona: str | None = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Cosine similarity search in pgvector."""
        table = self._table(embedding_model)
        conn = await self._get_conn()

        where_clause = ""
        params: list[Any] = [query_embedding, top_k]

        if persona:
            where_clause = "WHERE persona = %s"
            params = [query_embedding, persona, top_k]

        query = f"""
            SELECT content, persona, source_file, video_title,
                   video_date, video_url, chunk_index,
                   1 - (embedding <=> %s) AS similarity
            FROM {table}
            {where_clause}
            ORDER BY embedding <=> %s
            LIMIT %s
        """

        # Adjust params for the double reference to embedding
        if persona:
            params = [query_embedding, persona, query_embedding, top_k]
        else:
            params = [query_embedding, query_embedding, top_k]

        results: list[SearchResult] = []
        async with conn.cursor() as cur:
            await cur.execute(query, params)
            rows = await cur.fetchall()
            for rank, row in enumerate(rows):
                doc = Document(
                    content=row[0],
                    persona=row[1],
                    source_file=row[2],
                    video_title=row[3],
                    video_date=str(row[4]) if row[4] else "",
                    video_url=row[5],
                    chunk_index=row[6],
                )
                results.append(
                    SearchResult(document=doc, score=float(row[7]), rank=rank)
                )

        return results

    async def count(
        self, embedding_model: str, persona: str | None = None
    ) -> int:
        table = self._table(embedding_model)
        conn = await self._get_conn()

        if persona:
            query = f"SELECT COUNT(*) FROM {table} WHERE persona = %s"
            params: tuple = (persona,)
        else:
            query = f"SELECT COUNT(*) FROM {table}"
            params = ()

        async with conn.cursor() as cur:
            await cur.execute(query, params)
            row = await cur.fetchone()
            return row[0] if row else 0

    async def delete_by_persona(
        self, persona: str, embedding_model: str
    ) -> int:
        table = self._table(embedding_model)
        conn = await self._get_conn()

        async with conn.cursor() as cur:
            await cur.execute(
                f"DELETE FROM {table} WHERE persona = %s", (persona,)
            )
            return cur.rowcount

    async def close(self) -> None:
        if self._conn and not self._conn.closed:
            await self._conn.close()
            logger.info("pgvector connection closed")
