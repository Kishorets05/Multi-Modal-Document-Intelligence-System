"""Chunking data model.

A ``Chunk`` is the atomic unit produced by the chunking engine.  Every chunk
carries the full metadata required by downstream retrieval or indexing steps so
that a consumer never needs to re-open the source document.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """One semantically coherent unit of text with its full context.

    Attributes
    ----------
    document_id :
        Workspace identifier — matches the folder name under UPLOAD_DIR.
    chunk_id :
        Zero-based ordinal position within this document's chunk sequence.
    page_number :
        Page on which the chunk **starts** (0-based for PDF, 0 for text mode).
    heading :
        The nearest section heading that precedes this chunk, or ``""`` when
        the chunk appears before the first heading.
    document_type :
        The classification label from Module 4 (e.g. ``"resume"``).
    text :
        The chunk body text — never empty, never starts/ends with whitespace.
    token_count :
        Approximate word-token count (whitespace-split).  Exact enough for
        LLM context-window budgeting without requiring a tokeniser dependency.
    character_count :
        ``len(text)`` — cheap and deterministic.
    """

    document_id: str
    chunk_id: int
    page_number: int
    heading: str
    document_type: str
    text: str
    token_count: int = field(init=False)
    character_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.character_count = len(self.text)
        self.token_count = len(self.text.split())

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON responses."""
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "heading": self.heading,
            "document_type": self.document_type,
            "text": self.text,
            "token_count": self.token_count,
            "character_count": self.character_count,
        }
