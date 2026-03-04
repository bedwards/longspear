# Longspear — Ingest & Retrieval Comparison Report

## Data Coverage

| Persona | VTT Files | Chunks (nomic) | Chunks (mxbai) | Date Range |
|---------|-----------|----------------|----------------|------------|
| Heather Cox Richardson | 274 | 8,179 | 8,179 | Aug 2025 – Mar 2026 |
| Nate B Jones | 242+ | 5,165 | 5,165 | Aug 2025 – Mar 2026 |
| **Total** | **516+** | **13,344** | **13,344** | |

> NBJ download is ongoing — rate-limited by YouTube. 242 of ~800 videos fetched.

## Embedding Models

| Model | Dimensions | Size (VRAM) | Batch Speed (8K chunks) | Notes |
|-------|-----------|-------------|-------------------------|-------|
| `nomic-embed-text` | 768 | 0.5 GB | ~75s on Metal GPU | Faster, smaller vectors |
| `mxbai-embed-large` | 1024 | 0.7 GB | ~150s on Metal GPU | Higher quality, larger vectors |

Both models are permanently loaded (`keep_alive=-1`) in Ollama.

## Vector Stores

| Backend | Storage | Query Type | Index Type | Persistence |
|---------|---------|------------|------------|-------------|
| **pgvector** | PostgreSQL (Docker) | Server-based SQL | HNSW cosine | Docker volume |
| **LanceDB** | Embedded (in-process) | Python API | IVF_PQ default | Lance files on disk |

### Storage Matrix (all 4 combos populated)

| | pgvector | LanceDB |
|---|---------|---------|
| **nomic-embed-text** | 13,344 docs ✅ | 13,344 docs ✅ |
| **mxbai-embed-large** | 13,344 docs ✅ | 13,344 docs ✅ |

## Retrieval Quality Comparison

The system defaults to **pgvector + nomic-embed-text** for retrieval. Both backends use cosine similarity for ranking.

### Tradeoffs

| Factor | pgvector | LanceDB |
|--------|----------|---------|
| **Setup** | Requires PostgreSQL server | Zero-config embedded |
| **Scalability** | Industry standard, battle-tested | Excellent for prototyping |
| **Indexing** | HNSW with configurable params | Automatic IVF_PQ |
| **Filtering** | Full SQL WHERE clauses | Python filter expressions |
| **Persistence** | Transactional, ACID | File-based (Lance format) |
| **Concurrent access** | Multi-process safe | Single-process |

### Embedding Tradeoffs

| Factor | nomic-embed-text (768d) | mxbai-embed-large (1024d) |
|--------|------------------------|---------------------------|
| **Speed** | ~2x faster | Slower, more compute |
| **Quality** | Good general purpose | Higher semantic fidelity |
| **Memory** | Smaller vectors, less RAM | 33% larger vectors |
| **Use case** | Default for fast iteration | Potential quality upgrade |

## Architecture Choices

1. **Why dual stores?** — pgvector for production reliability, LanceDB for zero-config portability. Allows A/B testing retrieval quality.
2. **Why dual embeddings?** — nomic is fast for dev loops; mxbai may yield better retrieval for nuanced political topics. Both stored to enable comparison without re-embedding.
3. **Why not re-rank?** — The 123B Mistral model receives top-K chunks as context. Re-ranking adds latency with diminishing returns when the LLM can assess relevance itself.

## System Resources

| Resource | Value |
|----------|-------|
| Mistral-large:123b | 136.3 GB VRAM |
| nomic-embed-text | 0.5 GB VRAM |
| mxbai-embed-large | 0.7 GB VRAM |
| **Total GPU VRAM** | **137.6 GB** |
| Transcript disk | ~342 MB |
| LanceDB disk | ~75 MB |

All inference runs on native Ollama with Apple Metal GPU — no cloud, no Docker GPU passthrough.
