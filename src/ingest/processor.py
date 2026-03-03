"""Transcript processor: VTT → clean text → chunks.

Handles WebVTT subtitle files, strips formatting/timestamps,
and splits into overlapping chunks for embedding.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A processed text chunk ready for embedding."""

    text: str
    chunk_index: int
    source_file: str
    video_title: str = ""
    video_date: str = ""
    video_url: str = ""


class TranscriptProcessor:
    """Converts VTT subtitle files to clean text chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size

    def parse_vtt(self, vtt_path: Path) -> str:
        """Parse a WebVTT file into clean plain text.

        Strips timestamps, cue identifiers, and VTT formatting.
        Deduplicates overlapping auto-caption lines.
        """
        if not vtt_path.exists():
            logger.warning("VTT file not found: %s", vtt_path)
            return ""

        with open(vtt_path, encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Remove WEBVTT header and metadata
        content = re.sub(r"^WEBVTT.*?\n\n", "", content, flags=re.DOTALL)

        # Remove NOTE blocks
        content = re.sub(r"NOTE.*?\n\n", "", content, flags=re.DOTALL)

        # Remove timestamps (00:00:00.000 --> 00:00:05.000)
        content = re.sub(
            r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}.*?\n",
            "",
            content,
        )

        # Remove cue identifiers (numeric lines before timestamps)
        content = re.sub(r"^\d+\s*$", "", content, flags=re.MULTILINE)

        # Remove HTML tags (<c>, </c>, etc.)
        content = re.sub(r"<[^>]+>", "", content)

        # Remove position/alignment metadata
        content = re.sub(r"align:.*|position:.*|line:.*", "", content)

        # Collapse whitespace
        content = re.sub(r"\n{2,}", "\n", content)
        content = re.sub(r"[ \t]+", " ", content)

        # Deduplicate consecutive identical lines (common in auto-captions)
        lines = content.strip().split("\n")
        deduped: list[str] = []
        for line in lines:
            line = line.strip()
            if line and (not deduped or line != deduped[-1]):
                deduped.append(line)

        return " ".join(deduped)

    def chunk_text(
        self,
        text: str,
        source_file: str = "",
        video_title: str = "",
        video_date: str = "",
        video_url: str = "",
    ) -> list[TextChunk]:
        """Split text into overlapping chunks.

        Uses character-level chunking with overlap for context continuity.
        """
        if not text or len(text) < self._min_chunk_size:
            return []

        chunks: list[TextChunk] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self._chunk_size

            # Try to break at a sentence boundary
            if end < len(text):
                # Look for sentence-ending punctuation near the chunk boundary
                search_region = text[max(start, end - 100) : end + 100]
                for punct in [". ", "? ", "! ", ".\n", "?\n", "!\n"]:
                    last_punct = search_region.rfind(punct)
                    if last_punct != -1:
                        end = max(start, end - 100) + last_punct + len(punct)
                        break

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self._min_chunk_size:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        chunk_index=chunk_index,
                        source_file=source_file,
                        video_title=video_title,
                        video_date=video_date,
                        video_url=video_url,
                    )
                )
                chunk_index += 1

            # Move start forward, accounting for overlap
            start = end - self._chunk_overlap
            if start <= chunks[-1].chunk_index if chunks else 0:
                start = end  # Prevent infinite loop

            # Safety: ensure forward progress
            if start <= (end - self._chunk_size):
                start = end

        logger.debug(
            "Created %d chunks from %d chars (%s)",
            len(chunks), len(text), source_file,
        )
        return chunks

    def process_file(
        self,
        vtt_path: Path,
        video_title: str = "",
        video_date: str = "",
        video_url: str = "",
    ) -> list[TextChunk]:
        """Full pipeline: VTT file → clean text → chunks."""
        text = self.parse_vtt(vtt_path)
        if not text:
            return []

        return self.chunk_text(
            text=text,
            source_file=str(vtt_path.name),
            video_title=video_title,
            video_date=video_date,
            video_url=video_url,
        )
