"""Ingest pipeline: download → process → embed → store.

Orchestrates the full data ingestion for all configured channels,
embedding models, and vector store backends.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from ..config import get_settings, load_persona
from ..embeddings.base import create_embedding_provider
from ..vectorstores.base import Document, create_vector_store
from .downloader import TranscriptDownloader
from .processor import TranscriptProcessor

logger = logging.getLogger(__name__)


async def ingest_channel(
    channel_slug: str,
    embedding_models: list[str] | None = None,
    vectorstore_backends: list[str] | None = None,
    limit: int | None = None,
    skip_download: bool = False,
) -> dict[str, int]:
    """Ingest transcripts for a single channel.

    Args:
        channel_slug: Channel identifier (e.g. 'nate_b_jones').
        embedding_models: Models to use (default: all configured).
        vectorstore_backends: Backends to use (default: all configured).
        limit: Max videos to download (None = all).
        skip_download: Skip download, process existing files only.

    Returns:
        Dict of "{backend}_{model}" -> count of documents ingested.
    """
    settings = get_settings()
    channel = settings.channels.get(channel_slug)
    if not channel:
        raise ValueError(f"Unknown channel: {channel_slug}")

    # Defaults
    if embedding_models is None:
        embedding_models = list(settings.embedding.models.keys())
    if vectorstore_backends is None:
        vectorstore_backends = list(settings.vectorstore.backends.keys())

    # ── Step 1: Download transcripts ──────────────────────
    transcript_dir = settings.data_dir / "transcripts" / channel_slug
    downloader = TranscriptDownloader(
        output_dir=transcript_dir,
        cutoff_date=settings.data_cutoff_date,
    )

    if not skip_download:
        logger.info("Downloading transcripts for %s...", channel.name)
        videos = downloader.download_transcripts(
            channel_url=channel.url, limit=limit
        )
        logger.info("Downloaded %d transcripts for %s", len(videos), channel.name)
    else:
        videos = downloader._scan_downloaded()
        logger.info("Found %d existing transcripts for %s", len(videos), channel.name)

    if not videos:
        logger.warning("No transcripts found for %s", channel.name)
        return {}

    # ── Step 2: Process into chunks ───────────────────────
    processor = TranscriptProcessor(
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
        min_chunk_size=settings.chunking.min_chunk_size,
    )

    all_chunks = []
    for video in videos:
        if not video.subtitle_file or not video.subtitle_file.exists():
            continue

        chunks = processor.process_file(
            vtt_path=video.subtitle_file,
            video_title=video.title,
            video_date=video.upload_date,
            video_url=video.url,
        )
        all_chunks.extend(chunks)

    logger.info(
        "Processed %d chunks from %d videos for %s",
        len(all_chunks), len(videos), channel.name,
    )

    if not all_chunks:
        return {}

    # ── Step 3: Embed and store (for each model × backend) ──
    results: dict[str, int] = {}

    for model_name in embedding_models:
        logger.info("Embedding with %s...", model_name)
        embedder = create_embedding_provider(
            model_name=model_name,
            ollama_host=settings.ollama_host,
        )

        # Batch embed all chunks
        texts = [chunk.text for chunk in all_chunks]
        embeddings = await embedder.embed_batch(texts)

        # Build documents with embeddings
        documents = [
            Document(
                content=chunk.text,
                persona=channel_slug,
                source_file=chunk.source_file,
                video_title=chunk.video_title,
                video_date=chunk.video_date,
                video_url=chunk.video_url,
                chunk_index=chunk.chunk_index,
                embedding=emb,
            )
            for chunk, emb in zip(all_chunks, embeddings)
        ]

        # Store in each backend
        for backend_name in vectorstore_backends:
            logger.info(
                "Storing in %s (%s)...", backend_name, model_name
            )

            store_kwargs = {}
            if backend_name == "pgvector":
                store_kwargs["dsn"] = settings.postgres_dsn
            elif backend_name == "lancedb":
                lb_config = settings.vectorstore.backends.get("lancedb")
                store_kwargs["path"] = (
                    lb_config.path if lb_config else "./data/vectordb/lancedb"
                )

            store = create_vector_store(backend_name, **store_kwargs)
            await store.initialize()

            count = await store.add_documents(documents, model_name)
            results[f"{backend_name}_{model_name}"] = count

            await store.close()
            logger.info(
                "Stored %d documents in %s/%s",
                count, backend_name, model_name,
            )

    return results


async def ingest_all(
    limit: int | None = None,
    skip_download: bool = False,
) -> dict[str, dict[str, int]]:
    """Ingest all configured channels.

    Returns:
        Dict of channel_slug -> {backend_model: count}.
    """
    settings = get_settings()
    results: dict[str, dict[str, int]] = {}

    for slug in settings.channels:
        logger.info("═══ Ingesting channel: %s ═══", slug)
        results[slug] = await ingest_channel(
            channel_slug=slug,
            limit=limit,
            skip_download=skip_download,
        )

    return results


# ── CLI entrypoint ────────────────────────────────────────

async def _main() -> None:
    """CLI entrypoint for the ingest pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Longspear ingest pipeline")
    parser.add_argument(
        "--channel", type=str, help="Specific channel slug to ingest"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max videos per channel"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download, process existing files",
    )
    parser.add_argument(
        "--test-mode", action="store_true",
        help="Test mode: limit to 5 videos",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    limit = 5 if args.test_mode else args.limit

    if args.channel:
        results = {
            args.channel: await ingest_channel(
                channel_slug=args.channel,
                limit=limit,
                skip_download=args.skip_download,
            )
        }
    else:
        results = await ingest_all(
            limit=limit, skip_download=args.skip_download
        )

    print("\n═══ Ingest Results ═══")
    for channel, counts in results.items():
        print(f"\n  {channel}:")
        for key, count in counts.items():
            print(f"    {key}: {count} documents")


if __name__ == "__main__":
    asyncio.run(_main())
