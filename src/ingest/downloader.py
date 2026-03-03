"""YouTube transcript downloader using yt-dlp.

Downloads subtitles (manual or auto-generated) for all videos,
shorts, and livestreams from a channel, filtered by date.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Metadata for a downloaded video transcript."""

    video_id: str
    title: str
    upload_date: str  # YYYYMMDD
    url: str
    subtitle_file: Path | None = None
    video_type: str = "video"  # video, short, live


class TranscriptDownloader:
    """Downloads YouTube transcripts via yt-dlp.

    Handles videos, shorts, and livestreams. Filters by date.
    Prefers manual subtitles, falls back to auto-generated.
    """

    def __init__(self, output_dir: Path, cutoff_date: str) -> None:
        """
        Args:
            output_dir: Directory to save transcripts.
            cutoff_date: Only download videos after this date (YYYY-MM-DD).
        """
        self._output_dir = output_dir
        self._cutoff_date = cutoff_date
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _parse_cutoff(self) -> str:
        """Convert YYYY-MM-DD to YYYYMMDD for yt-dlp."""
        return self._cutoff_date.replace("-", "")

    def list_channel_videos(self, channel_url: str) -> list[dict]:
        """List all video metadata from a channel after cutoff date.

        Uses yt-dlp --flat-playlist to get metadata without downloading.
        """
        cutoff = self._parse_cutoff()
        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--print-json",
            "--dateafter", cutoff,
            "--no-warnings",
            channel_url,
        ]

        logger.info("Listing videos from %s after %s", channel_url, self._cutoff_date)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
        except subprocess.TimeoutExpired:
            logger.error("Timed out listing channel videos")
            return []

        videos = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                info = json.loads(line)
                videos.append(info)
            except json.JSONDecodeError:
                continue

        logger.info("Found %d videos after %s", len(videos), self._cutoff_date)
        return videos

    def download_transcripts(
        self,
        channel_url: str,
        limit: int | None = None,
    ) -> list[VideoInfo]:
        """Download transcripts for all matching videos from a channel.

        Args:
            channel_url: YouTube channel URL.
            limit: Max videos to download (None = all).

        Returns:
            List of VideoInfo for successfully downloaded transcripts.
        """
        cutoff = self._parse_cutoff()

        cmd = [
            "yt-dlp",
            # Subtitle options
            "--write-subs",
            "--write-auto-subs",
            "--sub-lang", "en",
            "--sub-format", "vtt",
            # Skip video download — we only want transcripts
            "--skip-download",
            # Date filter
            "--dateafter", cutoff,
            # Output template
            "-o", str(self._output_dir / "%(upload_date)s_%(id)s.%(ext)s"),
            # Write metadata
            "--write-info-json",
            # No warnings
            "--no-warnings",
            "--no-overwrites",
        ]

        if limit:
            cmd.extend(["--playlist-end", str(limit)])

        cmd.append(channel_url)

        logger.info(
            "Downloading transcripts from %s (limit=%s)", channel_url, limit
        )

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600
            )
            if result.returncode != 0:
                logger.warning("yt-dlp stderr: %s", result.stderr[:500])
        except subprocess.TimeoutExpired:
            logger.error("Timed out downloading transcripts")
            return []

        # Collect results
        return self._scan_downloaded()

    def _scan_downloaded(self) -> list[VideoInfo]:
        """Scan output directory for downloaded transcripts."""
        results: list[VideoInfo] = []

        for vtt_file in sorted(self._output_dir.glob("*.vtt")):
            # Parse filename: YYYYMMDD_VIDEOID.en.vtt
            stem = vtt_file.stem  # YYYYMMDD_VIDEOID.en
            parts = stem.split("_", 1)

            upload_date = parts[0] if len(parts) > 1 else ""
            video_id = parts[1].replace(".en", "") if len(parts) > 1 else stem

            # Try to load info json for metadata
            info_json = vtt_file.with_suffix("").with_suffix("").with_suffix(
                ".info.json"
            )
            title = ""
            url = ""
            video_type = "video"

            # Also check without the .en part
            alt_info = self._output_dir / f"{parts[0]}_{video_id.replace('.en', '')}.info.json"
            for candidate in [info_json, alt_info]:
                if candidate.exists():
                    try:
                        with open(candidate) as f:
                            info = json.load(f)
                            title = info.get("title", "")
                            url = info.get("webpage_url", "")
                            if info.get("duration", 0) and info["duration"] < 60:
                                video_type = "short"
                            if info.get("is_live"):
                                video_type = "live"
                    except (json.JSONDecodeError, KeyError):
                        pass
                    break

            results.append(
                VideoInfo(
                    video_id=video_id,
                    title=title,
                    upload_date=upload_date,
                    url=url,
                    subtitle_file=vtt_file,
                    video_type=video_type,
                )
            )

        logger.info("Found %d transcript files", len(results))
        return results
