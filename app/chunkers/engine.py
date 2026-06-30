"""Adaptive Intelligent Chunking Engine (Module 6).

The engine converts a document workspace into a sequence of semantically
coherent ``Chunk`` objects.  It reuses the existing ``LayoutEngine`` from
Module 5 so that:

* PDF documents benefit from real font-size and coordinate data.
* Non-PDF documents fall back to the same text heuristics already tested
  throughout Module 5.

Pipeline
--------
1. Try to build a ``LayoutEngine`` from the original PDF (workspace lookup).
2. If successful → ``group_into_paragraphs`` on the block list.
3. If not         → ``text_to_paragraphs`` on the plain-text string.
4. Run ``apply_overlap`` to enforce ``max_tokens`` budget with overlap seeding.
5. Wrap each result in a ``Chunk`` dataclass.

Configuration
-------------
``max_tokens``    (default 400) — soft upper bound per chunk.
``overlap_tokens``(default 50)  — words carried forward into the next chunk.
Both are tunable at call-time; no global state is mutated.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.chunkers.models import Chunk
from app.chunkers.utils import (
    apply_overlap,
    group_into_paragraphs,
    text_to_paragraphs,
)
from app.entity_extractors.layout_engine import LayoutEngine

logger = logging.getLogger("app")

# ── Defaults ────────────────────────────────────────────────────────────────
_DEFAULT_MAX_TOKENS: int = 400
_DEFAULT_OVERLAP_TOKENS: int = 50


class ChunkingEngine:
    """Convert a document workspace into an ordered list of ``Chunk`` objects.

    Parameters
    ----------
    workspace_dir :
        The document's workspace directory (contains ``original.pdf`` and/or
        ``extracted_text.txt``).
    document_id :
        Workspace identifier — stored verbatim in each ``Chunk``.
    document_type :
        Classification label from Module 4 — stored verbatim in each ``Chunk``.
    max_tokens :
        Soft per-chunk token limit.  A paragraph that exceeds this on its own
        is placed in its own chunk rather than being split.
    overlap_tokens :
        Number of words from the end of chunk N that seed the start of chunk
        N+1.  Set to 0 to disable overlap.
    """

    def __init__(
        self,
        workspace_dir: Path,
        document_id: str,
        document_type: str,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS,
    ) -> None:
        self._workspace_dir = Path(workspace_dir)
        self._document_id = document_id
        self._document_type = document_type
        self._max_tokens = max_tokens
        self._overlap_tokens = min(overlap_tokens, max_tokens // 2)

    # ── Public API ──────────────────────────────────────────────────────────

    def chunk(self, text: str) -> list[Chunk]:
        """Produce an ordered list of chunks for *text*.

        Args:
            text: Full plain-text content of the document (from
                  ``extracted_text.txt``).  Must be non-empty.

        Returns:
            A list of ``Chunk`` objects in reading order.  Returns ``[]``
            when *text* contains only whitespace.
        """
        if not text or not text.strip():
            logger.debug(
                "ChunkingEngine: empty text for document '%s' — returning [].",
                self._document_id,
            )
            return []

        paragraphs = self._extract_paragraphs(text)
        if not paragraphs:
            # Degenerate document: one block of text, no headings or blank lines.
            paragraphs = [(0, "", text.strip())]

        raw_chunks = apply_overlap(
            paragraphs,
            max_tokens=self._max_tokens,
            overlap_tokens=self._overlap_tokens,
        )

        chunks = [
            Chunk(
                document_id=self._document_id,
                chunk_id=idx,
                page_number=page,
                heading=heading,
                document_type=self._document_type,
                text=body,
            )
            for idx, (page, heading, body) in enumerate(raw_chunks)
        ]

        logger.debug(
            "ChunkingEngine: produced %d chunks for document '%s' "
            "(max_tokens=%d, overlap=%d).",
            len(chunks),
            self._document_id,
            self._max_tokens,
            self._overlap_tokens,
        )
        return chunks

    # ── Private helpers ─────────────────────────────────────────────────────

    def _extract_paragraphs(
        self, text: str
    ) -> list[tuple[int, str, str]]:
        """Return ``(page, heading, paragraph_text)`` triples.

        Attempts PDF layout extraction first; falls back to text heuristics.
        """
        engine = LayoutEngine.from_workspace(self._workspace_dir, text)

        if engine.source == "pdf" and engine.blocks:
            logger.debug(
                "ChunkingEngine: PDF layout mode — %d blocks for '%s'.",
                len(engine.blocks),
                self._document_id,
            )
            return group_into_paragraphs(engine.blocks, engine)

        # Text-heuristic fallback (non-PDF or empty PDF).
        logger.debug(
            "ChunkingEngine: text heuristic mode for '%s'.",
            self._document_id,
        )
        return text_to_paragraphs(text)
