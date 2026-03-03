"""Context builder: formats retrieved chunks into agent-ready prompts.

Takes retrieval results + persona config and builds the system
message that an agent (Claude Code, Gemini CLI) can use.
"""

from __future__ import annotations

import logging

from ..config import load_persona, PersonaConfig
from .retriever import RetrievalResponse

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds context-rich prompts from retrieved transcript chunks."""

    def build_system_prompt(
        self, persona_slug: str, retrieval: RetrievalResponse | None = None
    ) -> str:
        """Build a complete system prompt for persona impersonation.

        Args:
            persona_slug: Which persona to impersonate.
            retrieval: Optional retrieval results to ground the response.

        Returns:
            Complete system prompt string.
        """
        persona = load_persona(persona_slug)

        parts = [persona.system_prompt.strip()]

        # Add speaking style guidance
        if persona.speaking_style:
            parts.append("\n## Speaking Style")
            if "tone" in persona.speaking_style:
                parts.append(f"Tone: {persona.speaking_style['tone']}")
            if "patterns" in persona.speaking_style:
                parts.append("Speech patterns:")
                for p in persona.speaking_style["patterns"]:
                    parts.append(f"  - {p}")

        # Add biographical context
        if persona.biographical_context:
            parts.append(f"\n## Background\n{persona.biographical_context.strip()}")

        # Add retrieved transcript context
        if retrieval and retrieval.results:
            parts.append("\n## Recent Transcript Excerpts")
            parts.append(
                "The following are excerpts from your recent videos. "
                "Use these to ground your response in your actual recent commentary:"
            )
            for i, result in enumerate(retrieval.results, 1):
                doc = result.document
                header = f"\n### Excerpt {i}"
                if doc.video_title:
                    header += f" (from: {doc.video_title})"
                if doc.video_date:
                    header += f" [{doc.video_date}]"
                parts.append(header)
                parts.append(doc.content)
        else:
            parts.append(
                "\n## Note\n"
                "No specific transcript excerpts were retrieved for this query. "
                "Respond based on your general expertise, and acknowledge if "
                "you haven't recently covered this specific topic."
            )

        return "\n".join(parts)

    def build_debate_prompt(
        self,
        question: str,
        persona_slug: str,
        other_persona_response: str | None = None,
    ) -> str:
        """Build a user-turn prompt for the debate format.

        Args:
            question: The moderator's question.
            persona_slug: Which persona is responding.
            other_persona_response: Optional response from other panelist
                to react to.

        Returns:
            User prompt string.
        """
        parts = [f"Moderator: {question}"]

        if other_persona_response:
            parts.append(
                f"\n[The other panelist has just said the following. "
                f"You may respond to or build on their points:]\n"
                f"{other_persona_response}"
            )

        parts.append(
            f"\nPlease respond as yourself, drawing on the transcript "
            f"excerpts provided in your context when relevant."
        )

        return "\n".join(parts)
