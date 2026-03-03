"""Tests for the transcript processor."""

from pathlib import Path
from textwrap import dedent

from src.ingest.processor import TranscriptProcessor


def _write_vtt(tmp_path: Path, content: str) -> Path:
    """Write a VTT file to tmp and return its path."""
    vtt = tmp_path / "test.vtt"
    vtt.write_text(content)
    return vtt


def test_parse_vtt_strips_timestamps(tmp_path: Path):
    """VTT parser removes timestamps and cue IDs."""
    vtt_content = dedent("""\
        WEBVTT

        1
        00:00:00.000 --> 00:00:05.000
        Hello, welcome to the show.

        2
        00:00:05.000 --> 00:00:10.000
        Today we're discussing AI regulation.
    """)
    vtt = _write_vtt(tmp_path, vtt_content)
    processor = TranscriptProcessor()
    text = processor.parse_vtt(vtt)

    assert "Hello, welcome to the show." in text
    assert "AI regulation" in text
    assert "-->" not in text
    assert "WEBVTT" not in text


def test_parse_vtt_deduplicates(tmp_path: Path):
    """Auto-caption duplicate lines are removed."""
    vtt_content = dedent("""\
        WEBVTT

        00:00:00.000 --> 00:00:02.000
        This is a test.

        00:00:02.000 --> 00:00:04.000
        This is a test.

        00:00:04.000 --> 00:00:06.000
        Another sentence.
    """)
    vtt = _write_vtt(tmp_path, vtt_content)
    processor = TranscriptProcessor()
    text = processor.parse_vtt(vtt)

    # Should have only one instance of "This is a test."
    assert text.count("This is a test.") == 1
    assert "Another sentence." in text


def test_chunk_text_creates_chunks():
    """Chunking produces expected number of chunks."""
    processor = TranscriptProcessor(
        chunk_size=100, chunk_overlap=20, min_chunk_size=10
    )
    text = "Word " * 200  # 1000 chars
    chunks = processor.chunk_text(text, source_file="test.vtt")

    assert len(chunks) > 1
    assert all(c.source_file == "test.vtt" for c in chunks)
    assert chunks[0].chunk_index == 0


def test_chunk_text_respects_min_size():
    """Chunks below min_chunk_size are discarded."""
    processor = TranscriptProcessor(
        chunk_size=100, chunk_overlap=20, min_chunk_size=50
    )
    text = "Short."
    chunks = processor.chunk_text(text)

    assert len(chunks) == 0


def test_process_file_end_to_end(tmp_path: Path):
    """Full pipeline: VTT → clean text → chunks."""
    vtt_content = dedent("""\
        WEBVTT

        00:00:00.000 --> 00:00:10.000
        This is the first segment of a longer transcript about artificial intelligence
        and its impact on modern society, covering topics from machine learning to ethics.

        00:00:10.000 --> 00:00:20.000
        We need to consider the implications of deploying these systems at scale,
        particularly in areas like healthcare, education, and criminal justice.
    """)
    vtt = _write_vtt(tmp_path, vtt_content)
    processor = TranscriptProcessor(
        chunk_size=100, chunk_overlap=20, min_chunk_size=50
    )
    chunks = processor.process_file(
        vtt, video_title="Test Video", video_date="20250901"
    )

    assert len(chunks) > 0
    assert chunks[0].video_title == "Test Video"
    assert chunks[0].video_date == "20250901"
