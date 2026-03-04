"""FastAPI server for the Longspear RAG system.

Exposes endpoints for agents (Claude Code, Gemini CLI) to:
  - Query for context-augmented persona prompts
  - Check system health and document stats
  - Trigger data ingestion
  - Chat with personas via streaming LLM responses
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import get_settings, list_personas, load_persona
from ..conversation.engine import ConversationEngine, ChatMessage
from ..embeddings.base import create_embedding_provider
from ..retrieval.context_builder import ContextBuilder
from ..retrieval.retriever import Retriever, RetrievalRequest
from ..vectorstores.base import create_vector_store

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Longspear RAG API",
    description=(
        "Local RAG system for AI-moderated debates. "
        "Retrieves transcript context and builds persona prompts."
    ),
    version="0.2.0",
)

# Serve static files (web UI)
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ── Request/Response Models ───────────────────────────────

class QueryRequest(BaseModel):
    """Request to query the RAG system."""

    question: str = Field(..., description="The question or topic to discuss")
    persona: str = Field(..., description="Persona slug (e.g. 'nate_b_jones')")
    vectorstore: str | None = Field(None, description="Vector store backend")
    embedding: str | None = Field(None, description="Embedding model name")
    top_k: int | None = Field(None, description="Number of chunks to retrieve")
    other_response: str | None = Field(
        None, description="Other panelist's response (for debate format)"
    )


class QueryResponse(BaseModel):
    """Response from the RAG system."""

    system_prompt: str
    user_prompt: str
    persona: str
    vectorstore_used: str
    embedding_used: str
    chunks_retrieved: int
    sources: list[dict[str, str]]


class HealthResponse(BaseModel):
    status: str
    services: dict[str, str]


class StatsResponse(BaseModel):
    counts: dict[str, dict[str, int]]


class IngestRequest(BaseModel):
    channel: str | None = Field(None, description="Specific channel slug")
    limit: int | None = Field(None, description="Max videos per channel")
    skip_download: bool = Field(False, description="Skip downloading, process existing")


class IngestResponse(BaseModel):
    results: dict[str, dict[str, int]]


class ChatRequest(BaseModel):
    """Request to chat with a persona."""

    question: str = Field(..., description="Your question or topic")
    persona: str = Field(..., description="Persona slug")
    vectorstore: str | None = Field(None, description="Vector store backend")
    embedding: str | None = Field(None, description="Embedding model name")
    top_k: int | None = Field(None, description="Number of chunks to retrieve")
    history: list[dict[str, str]] | None = Field(
        None, description="Conversation history [{role, content}]"
    )
    stream: bool = Field(True, description="Stream response via SSE")


class DebateRequest(BaseModel):
    """Request a moderated debate between two personas."""

    question: str = Field(..., description="The moderator's question")
    persona_a: str = Field(
        "heather_cox_richardson", description="First persona slug"
    )
    persona_b: str = Field("nate_b_jones", description="Second persona slug")
    vectorstore: str | None = Field(None)
    embedding: str | None = Field(None)
    top_k: int | None = Field(None)


# ── Endpoints ─────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Serve the web UI."""
    index_path = _STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(
        content="<h1>Longspear</h1><p>Web UI not found. "
        "Check src/api/static/index.html</p>",
        status_code=200,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check health of all services."""
    settings = get_settings()
    services: dict[str, str] = {}

    # Check Ollama
    try:
        import ollama as ollama_client
        client = ollama_client.AsyncClient(host=settings.ollama_host)
        await client.list()
        services["ollama"] = "healthy"
    except Exception as e:
        services["ollama"] = f"unhealthy: {e}"

    # Check PostgreSQL
    try:
        import psycopg
        conn = await psycopg.AsyncConnection.connect(
            settings.postgres_dsn, autocommit=True
        )
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")
        await conn.close()
        services["postgres"] = "healthy"
    except Exception as e:
        services["postgres"] = f"unhealthy: {e}"

    # LanceDB is always available (embedded)
    services["lancedb"] = "healthy"

    overall = "healthy" if all(
        v == "healthy" for v in services.values()
    ) else "degraded"

    return HealthResponse(status=overall, services=services)


@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat with a persona. Streams response via SSE by default."""
    # Validate persona
    try:
        load_persona(request.persona)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Persona '{request.persona}' not found. "
                   f"Available: {list_personas()}",
        )

    engine = ConversationEngine()

    # Convert history
    history = None
    if request.history:
        history = [
            ChatMessage(role=h["role"], content=h["content"])
            for h in request.history
        ]

    if request.stream:
        # Streaming SSE response
        async def event_stream():
            try:
                # First, send metadata
                ctx = await engine.prepare_context(
                    question=request.question,
                    persona_slug=request.persona,
                    embedding_model=request.embedding,
                    vectorstore_backend=request.vectorstore,
                    top_k=request.top_k,
                )
                meta = {
                    "type": "meta",
                    "persona": request.persona,
                    "embedding": ctx.embedding_used,
                    "vectorstore": ctx.vectorstore_used,
                    "chunks_retrieved": ctx.chunks_retrieved,
                    "sources": ctx.sources,
                }
                yield f"data: {json.dumps(meta)}\n\n"

                # Then stream tokens
                async for token in engine.chat_stream(
                    question=request.question,
                    persona_slug=request.persona,
                    history=history,
                    embedding_model=request.embedding,
                    vectorstore_backend=request.vectorstore,
                    top_k=request.top_k,
                ):
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            except Exception as e:
                logger.exception("Chat stream error")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    else:
        # Non-streaming response
        response = await engine.chat(
            question=request.question,
            persona_slug=request.persona,
            history=history,
            embedding_model=request.embedding,
            vectorstore_backend=request.vectorstore,
            top_k=request.top_k,
        )
        return {"persona": request.persona, "response": response}


@app.post("/debate")
async def debate(request: DebateRequest):
    """Run a moderated debate between two personas. Streams via SSE."""
    # Validate both personas
    for p in [request.persona_a, request.persona_b]:
        try:
            load_persona(p)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Persona '{p}' not found. Available: {list_personas()}",
            )

    engine = ConversationEngine()

    async def debate_stream():
        try:
            # Persona A goes first
            yield f"data: {json.dumps({'type': 'turn_start', 'persona': request.persona_a})}\n\n"

            response_a_parts: list[str] = []
            async for token in engine.chat_stream(
                question=request.question,
                persona_slug=request.persona_a,
                embedding_model=request.embedding,
                vectorstore_backend=request.vectorstore,
                top_k=request.top_k,
            ):
                response_a_parts.append(token)
                yield f"data: {json.dumps({'type': 'token', 'persona': request.persona_a, 'content': token})}\n\n"

            response_a = "".join(response_a_parts)
            yield f"data: {json.dumps({'type': 'turn_end', 'persona': request.persona_a})}\n\n"

            # Persona B responds, given A's response
            yield f"data: {json.dumps({'type': 'turn_start', 'persona': request.persona_b})}\n\n"

            async for token in engine.chat_stream(
                question=request.question,
                persona_slug=request.persona_b,
                embedding_model=request.embedding,
                vectorstore_backend=request.vectorstore,
                top_k=request.top_k,
                other_response=response_a,
            ):
                yield f"data: {json.dumps({'type': 'token', 'persona': request.persona_b, 'content': token})}\n\n"

            yield f"data: {json.dumps({'type': 'turn_end', 'persona': request.persona_b})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.exception("Debate stream error")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        debate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Query the RAG system for a persona-grounded response context.

    Returns the system prompt and user prompt that an agent should use.
    """
    # Validate persona exists
    try:
        persona = load_persona(request.persona)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Persona '{request.persona}' not found. "
                   f"Available: {list_personas()}",
        )

    # Retrieve relevant context
    retriever = Retriever()
    retrieval = await retriever.retrieve(
        RetrievalRequest(
            question=request.question,
            persona=request.persona,
            embedding_model=request.embedding,
            vectorstore_backend=request.vectorstore,
            top_k=request.top_k,
        )
    )

    # Build prompts
    builder = ContextBuilder()
    system_prompt = builder.build_system_prompt(
        persona_slug=request.persona,
        retrieval=retrieval,
    )
    user_prompt = builder.build_debate_prompt(
        question=request.question,
        persona_slug=request.persona,
        other_persona_response=request.other_response,
    )

    # Collect source references
    sources = []
    for r in retrieval.results:
        source = {}
        if r.document.video_title:
            source["title"] = r.document.video_title
        if r.document.video_date:
            source["date"] = r.document.video_date
        if r.document.video_url:
            source["url"] = r.document.video_url
        source["score"] = f"{r.score:.4f}"
        sources.append(source)

    return QueryResponse(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        persona=request.persona,
        vectorstore_used=retrieval.vectorstore_backend,
        embedding_used=retrieval.embedding_model,
        chunks_retrieved=len(retrieval.results),
        sources=sources,
    )


@app.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    """Get document counts per vector store and embedding model."""
    settings = get_settings()
    counts: dict[str, dict[str, int]] = {}

    for backend_name in settings.vectorstore.backends:
        store_kwargs: dict[str, Any] = {}
        if backend_name == "pgvector":
            store_kwargs["dsn"] = settings.postgres_dsn
        elif backend_name == "lancedb":
            lb_config = settings.vectorstore.backends.get("lancedb")
            store_kwargs["path"] = (
                lb_config.path if lb_config else "./data/vectordb/lancedb"
            )

        try:
            store = create_vector_store(backend_name, **store_kwargs)
            await store.initialize()

            backend_counts: dict[str, int] = {}
            for model_name in settings.embedding.models:
                try:
                    backend_counts[model_name] = await store.count(model_name)
                except Exception:
                    backend_counts[model_name] = -1

            counts[backend_name] = backend_counts
            await store.close()
        except Exception as e:
            counts[backend_name] = {"error": -1}

    return StatsResponse(counts=counts)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    """Trigger data ingestion pipeline."""
    from ..ingest.pipeline import ingest_all, ingest_channel

    if request.channel:
        results = {
            request.channel: await ingest_channel(
                channel_slug=request.channel,
                limit=request.limit,
                skip_download=request.skip_download,
            )
        }
    else:
        results = await ingest_all(
            limit=request.limit,
            skip_download=request.skip_download,
        )

    return IngestResponse(results=results)


@app.get("/personas")
async def get_personas() -> list[dict[str, str]]:
    """List available personas."""
    slugs = list_personas()
    result = []
    for slug in slugs:
        try:
            persona = load_persona(slug)
            result.append({"slug": slug, "name": persona.name})
        except Exception:
            result.append({"slug": slug, "name": slug})
    return result

