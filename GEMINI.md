# Longspear — Project Context

## Architecture
Local Docker-based RAG system for AI-moderated debates. Impersonates **Heather Cox Richardson** and **Nate B Jones** using YouTube transcripts (post Aug 2025, Claude Opus 4's training cutoff).

- **Inference**: Native Ollama with Metal GPU (not Docker) — `mistral-large:123b`, `nomic-embed-text`, `mxbai-embed-large`
- **Storage**: Dual vector stores (pgvector + LanceDB), dual embeddings (nomic + mxbai)
- **App**: Python/FastAPI + PostgreSQL in Docker, config-driven, SOLID/DRY/APIE

## Conversation Pipeline
1. Embed question with `nomic-embed-text` via Ollama
2. Retrieve relevant transcript chunks from pgvector (cosine similarity)
3. Build persona-grounded prompt with `ContextBuilder`
4. Generate response via `mistral-large:123b` through Ollama (Metal GPU)
5. Stream tokens to browser/TUI via SSE

## Key Components
| Module | Purpose |
|--------|---------|
| `src/conversation/engine.py` | RAG → Mistral streaming orchestrator |
| `src/api/server.py` | FastAPI with `/chat`, `/debate`, `/query`, `/ingest` |
| `src/api/static/` | Web UI (Debate, Chat, Monitor modes) |
| `src/embeddings/` | Ollama embedding providers (nomic, mxbai) |
| `src/vectorstores/` | pgvector + LanceDB backends |
| `src/ingest/` | yt-dlp downloader + VTT parser + chunker |
| `src/retrieval/` | Retriever + context builder |
| `config/personas/` | YAML persona definitions (system prompts, styles) |
| `scripts/tui.py` | Standalone terminal UI (ANSI, no deps) |
| `scripts/setup.sh` | Docker management + monitor command |

## Ports
- `11434` — Native Ollama (host)
- `25432` → `5432` — PostgreSQL (Docker)
- `28000` → `8000` — FastAPI app (Docker)

## Data
- Transcripts: `data/transcripts/{persona}/*.vtt`
- LanceDB: `data/vectordb/lancedb/`
- Config: `config/settings.yaml`, `config/personas/*.yaml`
