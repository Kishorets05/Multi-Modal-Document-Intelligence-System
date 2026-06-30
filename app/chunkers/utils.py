"""Reusable chunking utilities.

These helpers operate on lists of ``LayoutBlock`` objects produced by the
existing ``LayoutEngine`` (Module 5) and on plain-text strings when no PDF
is available.  They are deliberately stateless functions so that the engine
class stays thin and testable.

Public API
----------
``estimate_tokens(text)``      — cheap word-token approximation.
``is_heading_block(block, engine)`` — delegate to LayoutEngine heading logic.
``group_into_paragraphs(blocks)``   — merge adjacent non-heading blocks.
``split_overlapping(chunks, max_tokens, overlap_tokens)``
    — sliding-window overlap pass over a finished chunk list.
``text_to_paragraphs(text)``   — fallback for plain-text mode.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.entity_extractors.layout_engine import LayoutBlock, LayoutEngine


# ─────────────────────────────────────────────────────────────────────────── #
#  Token estimation                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def estimate_tokens(text: str) -> int:
    """Return the approximate word-token count for *text*.

    Uses whitespace splitting — fast, deterministic, and good enough for
    context-window budgeting without adding a tokeniser dependency.
    """
    return len(text.split())


# ─────────────────────────────────────────────────────────────────────────── #
#  Heading detection                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

_KNOWN_HEADING_KEYWORDS = {
    # Resumes
    "HEADER", "CONTACT", "SUMMARY", "EXECUTIVE SUMMARY", "OBJECTIVE", "PROFILE", 
    "EDUCATION", "SKILLS", "EXPERIENCE", "INTERNSHIP", "PROJECTS", "CERTIFICATIONS", 
    "ACHIEVEMENTS", "PUBLICATIONS", "REFERENCES", "LANGUAGES", "EMPLOYMENT", 
    "BACKGROUND", "COMPETENCIES", "ACTIVITIES", "INTERESTS", "AWARDS", "HONORS", "VOLUNTEER",
    # Invoice
    "VENDOR INFORMATION", "CUSTOMER INFORMATION", "INVOICE DETAILS", "LINE ITEMS", 
    "PAYMENT SUMMARY", "PAYMENT TERMS", "BILL TO", "SHIP TO", "TOTAL",
    # Contract
    "PARTIES", "DEFINITIONS", "SCOPE", "RESPONSIBILITIES", "CONFIDENTIALITY", 
    "LIABILITY", "TERMINATION", "GOVERNING LAW", "SIGNATURES", "RECITALS", 
    "TERMS AND CONDITIONS", "WITNESS", "OBLIGATIONS",
    # Medical
    "PATIENT INFORMATION", "PATIENT DETAILS", "CLINICAL HISTORY", "DIAGNOSIS", 
    "EXAMINATION", "LAB RESULTS", "MEDICATIONS", "TREATMENT PLAN", "RECOMMENDATIONS", "PLAN",
    # Research / Academic
    "TITLE", "AUTHORS", "ABSTRACT", "INTRODUCTION", "LITERATURE REVIEW", 
    "METHODOLOGY", "RESULTS", "DISCUSSION", "CONCLUSION", "ACKNOWLEDGEMENTS", 
    "APPENDIX", "MAIN SECTIONS"
}

def _is_known_heading(text: str) -> bool:
    s = text.upper().strip()
    # Strip leading numbers/letters for numbered lists (e.g. "1. INTRODUCTION", "II. METHODOLOGY", "A) BACKGROUND")
    s = re.sub(r'^(?:[0-9]+|[IVX]+|[A-Z])[\.\-\)]\s*', '', s)
    s = re.sub(r'[^\w\s]', '', s)
    s = " ".join(s.split())
    # Match exact, or compound headings
    for kw in _KNOWN_HEADING_KEYWORDS:
        if s == kw or s.startswith(kw + " ") or s.endswith(" " + kw):
            return True
    return False

def is_heading_block(block: "LayoutBlock", engine: "LayoutEngine") -> bool:
    """Determine if a block is a major document section heading.
    
    Refined logic:
    - Avoids treating ALL-CAPS project titles or company names as sections.
    - Uses a known vocabulary of major sections.
    - In PDF mode, strongly formatted text (significantly larger font) is also accepted.
    """
    if block.is_empty:
        return False
        
    is_known = _is_known_heading(block.stripped)
    
    if engine.source == "pdf":
        # Known section + basic formatting (bold/caps)
        if is_known and engine.is_heading(block):
            return True
        # Unknown section, but has significantly larger font size than average
        if block.font_size > engine._avg_font_size * 1.15:
            return True
        return False
    else:
        # Text-heuristic mode
        return is_known and engine.is_heading(block)


# ─────────────────────────────────────────────────────────────────────────── #
#  Paragraph grouping                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def group_into_paragraphs(
    blocks: list["LayoutBlock"],
    engine: "LayoutEngine",
) -> list[tuple[int, str, str]]:
    """Group adjacent non-heading blocks into paragraphs.

    Headings act as paragraph separators.  Content blocks between two
    headings are accumulated into one paragraph per blank-line gap.
    The heading itself is emitted as the first paragraph of the new section
    so that its semantic text is preserved in the chunk body.

    Returns
    -------
    list of ``(page_number, heading, paragraph_text)`` tuples in reading order.
    """
    paragraphs: list[tuple[int, str, str]] = []
    current_heading: str = ""
    current_page: int = 0
    buffer: list[str] = []
    buffer_page: int = 0

    def _flush() -> None:
        joined = " ".join(buffer).strip()
        if joined:
            paragraphs.append((buffer_page, current_heading, joined))
        buffer.clear()

    for block in blocks:
        if block.is_empty:
            # Blank line → paragraph boundary inside a section.
            _flush()
            continue

        if is_heading_block(block, engine):
            _flush()
            current_heading = block.stripped
            current_page = block.page_num
            # Emit the heading as its own paragraph so it appears in the chunk text.
            paragraphs.append((current_page, current_heading, current_heading))
            continue

        if not buffer:
            buffer_page = block.page_num

        buffer.append(block.stripped)

    _flush()
    return paragraphs


# ─────────────────────────────────────────────────────────────────────────── #
#  Overlap                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

def apply_overlap(
    paragraphs: list[tuple[int, str, str]],
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, str, str]]:
    """Merge paragraphs into semantically coherent chunks.

    Rules
    -----
    * A chunk should ideally represent a semantic section (determined by heading).
    * If a section is very small (e.g. < ~150 tokens), it is merged into the
      adjacent chunk to avoid tiny isolated chunks.
    * A paragraph is never split — it is always placed in its entirety.
    * If a chunk exceeds *max_tokens*, it is sealed. The new chunk is seeded
      with the last ``overlap_tokens`` words from the previous chunk.
    * Reading order is strictly preserved.
    """
    if not paragraphs:
        return []

    chunks: list[tuple[int, str, str]] = []
    chunk_page: int = paragraphs[0][0]
    chunk_heading: str = paragraphs[0][1]
    chunk_words: list[str] = []

    # Threshold below which we don't seal a semantic chunk (we merge it).
    # We only merge extremely small sections (approx < 45 tokens) to avoid tiny isolated chunks.
    # Otherwise, every major section boundary strictly starts a new chunk.
    min_tokens_to_seal = 45

    def _seal() -> None:
        text = " ".join(chunk_words).strip()
        if text:
            chunks.append((chunk_page, chunk_heading, text))

    for page, heading, text in paragraphs:
        para_words = text.split()
        projected_len = len(chunk_words) + len(para_words)

        # It's a new section if we have a heading, AND it's different from the chunk's current heading.
        is_new_section = bool(heading and heading != chunk_heading)

        # 1. Semantic boundary: New section, and current chunk is large enough.
        if is_new_section and len(chunk_words) >= min_tokens_to_seal:
            _seal()
            # Reset overlap completely on semantic boundaries. Do not leak context.
            chunk_words = []
            chunk_page = page
            chunk_heading = heading
            projected_len = len(chunk_words) + len(para_words)
            is_new_section = False

        # 2. Token limit: Adding this paragraph exceeds max_tokens.
        # (An oversized paragraph is kept whole).
        if chunk_words and projected_len > max_tokens:
            _seal()
            # Preserve local context via overlap when splitting a single massive section.
            overlap_seed = chunk_words[-overlap_tokens:] if overlap_tokens > 0 else []
            chunk_words = list(overlap_seed)
            chunk_page = page
            # Keep the new heading if this paragraph was the start of a new section
            # that was too small to trigger the semantic split, or keep the old one.
            chunk_heading = heading
            
        # Update heading tracking when entering a new section but we didn't seal
        # (e.g. intro text -> first heading, or merging a tiny section).
        if heading and heading != chunk_heading and not chunk_words:
            chunk_heading = heading
            chunk_page = page
        elif heading and not chunk_heading:
            # If chunk is currently accumulating intro text and hits the first heading,
            # adopt the heading so the chunk represents this section.
            chunk_heading = heading

        chunk_words.extend(para_words)

    _seal()
    return chunks


# ─────────────────────────────────────────────────────────────────────────── #
#  Plain-text fallback                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

# Heading heuristics for text-only mode (mirrors LayoutEngine text heuristics).
_ALLCAPS_HEAD_RE = re.compile(r"^[A-Z][A-Z\s\-/&()+]{1,70}$")
_TITLECASE_HEAD_RE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]*){0,6}:?\s*$")


def _text_is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80:
        return False
        
    is_known = _is_known_heading(s)
    
    looks_like_heading = bool(_ALLCAPS_HEAD_RE.match(s)) or (
        bool(_TITLECASE_HEAD_RE.match(s)) and not s.endswith(".")
    )
    
    return is_known and looks_like_heading


def text_to_paragraphs(text: str) -> list[tuple[int, str, str]]:
    """Convert a plain-text string to ``(page, heading, paragraph)`` triples.

    Used when no PDF is available and the ``LayoutEngine`` operates in
    text-heuristic mode.  Paragraph boundaries are inferred from blank lines;
    headings are detected with the same heuristics as ``LayoutEngine``.
    """
    paragraphs: list[tuple[int, str, str]] = []
    current_heading: str = ""
    buffer: list[str] = []

    def _flush() -> None:
        joined = " ".join(buffer).strip()
        if joined:
            paragraphs.append((0, current_heading, joined))
        buffer.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            _flush()
            continue
        if _text_is_heading(line):
            _flush()
            current_heading = line
            # Treat the heading as its own paragraph so it's not lost
            paragraphs.append((0, current_heading, current_heading))
            continue
        buffer.append(line)

    _flush()
    return paragraphs
