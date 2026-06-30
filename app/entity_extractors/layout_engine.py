"""Layout extraction engine — Hybrid Layout-Aware pipeline.

When a PDF is available in the document workspace the engine uses
PyMuPDF's ``page.get_text("dict")`` to obtain:

* bounding-box coordinates  (x0, y0, x1, y1)
* font size per span
* bold / italic flags per span
* reading order (preserved by PyMuPDF sort=True)
* page number

When the original file is not a PDF, or PyMuPDF cannot open it, the engine
falls back to **heuristic text-layout analysis**:

* ALL-CAPS short lines → treated as large-font headings
* Indented lines       → treated as body content
* Blank lines          → preserved as separators

All downstream consumers (SectionParser, document parsers) work with the
same ``list[LayoutBlock]`` regardless of which mode was used.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("app")

# ─────────────────────────────────────────────────────────────────────────── #
#  Data model                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class LayoutBlock:
    """One line of text with positional and typographic metadata."""

    text: str
    page_num: int = 0
    x0: float = 0.0          # left edge
    y0: float = 0.0          # top edge
    x1: float = 0.0          # right edge
    y1: float = 0.0          # bottom edge
    font_size: float = 12.0
    is_bold: bool = False
    is_italic: bool = False
    block_index: int = 0     # absolute position in reading order

    @property
    def stripped(self) -> str:
        return self.text.strip()

    @property
    def is_empty(self) -> bool:
        return not self.stripped

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LayoutBlock(idx={self.block_index}, "
            f"size={self.font_size:.1f}, bold={self.is_bold}, "
            f"text={self.stripped[:40]!r})"
        )


# ─────────────────────────────────────────────────────────────────────────── #
#  Text-mode heuristics                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

# Matches lines that are plausibly section headings in text-only mode.
_ALLCAPS_HEAD_RE = re.compile(r"^[A-Z][A-Z\s\-/&()+]{1,70}$")
_TITLECASE_HEAD_RE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]*){0,6}:?\s*$")


def _text_infer_size(line: str) -> float:
    s = line.strip()
    if not s:
        return 10.0
    if _ALLCAPS_HEAD_RE.match(s):
        return 16.0
    if _TITLECASE_HEAD_RE.match(s) and len(s) <= 50:
        return 14.0
    return 12.0


def _text_infer_bold(line: str) -> bool:
    return bool(_ALLCAPS_HEAD_RE.match(line.strip()))


def _text_looks_like_heading(s: str) -> bool:
    """Fast heuristic heading check used in text-only mode."""
    if not s or len(s) > 80:
        return False
    return bool(_ALLCAPS_HEAD_RE.match(s)) or (
        bool(_TITLECASE_HEAD_RE.match(s)) and not s.endswith(".")
    )


# ─────────────────────────────────────────────────────────────────────────── #
#  Layout Engine                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

_PDF_SUFFIXES = {".pdf"}


class LayoutEngine:
    """Produces a flat, reading-ordered list of ``LayoutBlock`` objects.

    Construction
    ------------
    Use the class-method factories rather than calling ``__init__`` directly::

        engine = LayoutEngine.from_workspace(workspace_dir, fallback_text)
        engine = LayoutEngine.from_pdf(path_to_pdf)
        engine = LayoutEngine.from_text(plain_text_string)
    """

    def __init__(self, blocks: list[LayoutBlock], source: str = "text") -> None:
        self._blocks: list[LayoutBlock] = blocks
        self.source: str = source   # "pdf" | "text"
        self._avg_font_size: float = self._calc_avg()
        # Heading threshold: a block is a heading if its font is ≥ threshold.
        self._heading_threshold: float = self._avg_font_size * 1.12

    # ── factories ────────────────────────────────────────────────────────── #

    @classmethod
    def from_workspace(
        cls,
        workspace_dir: Path,
        fallback_text: str,
    ) -> "LayoutEngine":
        """Try to build from the original PDF; fall back to plain text."""
        pdf_path = cls._find_pdf(workspace_dir)
        if pdf_path is not None:
            engine = cls.from_pdf(pdf_path)
            if engine.source == "pdf":
                return engine
        logger.debug(
            "LayoutEngine: using text-only mode for workspace '%s'.",
            workspace_dir.name,
        )
        return cls.from_text(fallback_text)

    @classmethod
    def from_pdf(cls, pdf_path: Path) -> "LayoutEngine":
        """Build from PDF using ``page.get_text('dict')``.

        Each *line* in each PyMuPDF block becomes one ``LayoutBlock``.
        Font size and bold/italic flags are taken from the dominant span.
        Coordinates and page number are preserved verbatim.
        """
        try:
            import fitz
        except ImportError:
            logger.warning(
                "PyMuPDF (fitz) not available — falling back to text mode."
            )
            return cls([], source="pdf_unavailable")

        blocks: list[LayoutBlock] = []
        idx = 0
        try:
            doc = fitz.open(str(pdf_path))
            for page_num, page in enumerate(doc):
                # sort=True preserves natural reading order
                page_dict = page.get_text("dict", sort=True)
                for raw_block in page_dict.get("blocks", []):
                    if raw_block.get("type") != 0:
                        continue  # skip image blocks

                    for line in raw_block.get("lines", []):
                        text_parts: list[str] = []
                        max_size: float = 0.0
                        is_bold = False
                        is_italic = False
                        bbox = line.get("bbox", (0.0, 0.0, 0.0, 0.0))

                        for span in line.get("spans", []):
                            span_txt = span.get("text", "")
                            if span_txt.strip():
                                text_parts.append(span_txt)
                            fs = float(span.get("size", 12.0))
                            if fs > max_size:
                                max_size = fs
                            flags: int = span.get("flags", 0)
                            if flags & 16:   # bold bit
                                is_bold = True
                            if flags & 2:    # italic bit
                                is_italic = True

                        line_text = "".join(text_parts)
                        if not line_text.strip():
                            continue

                        blocks.append(LayoutBlock(
                            text=line_text,
                            page_num=page_num,
                            x0=bbox[0],
                            y0=bbox[1],
                            x1=bbox[2],
                            y1=bbox[3],
                            font_size=max_size if max_size > 0 else 12.0,
                            is_bold=is_bold,
                            is_italic=is_italic,
                            block_index=idx,
                        ))
                        idx += 1

            doc.close()
            logger.debug(
                "LayoutEngine: extracted %d blocks from PDF '%s'.",
                len(blocks),
                pdf_path.name,
            )
        except Exception as exc:
            logger.warning(
                "LayoutEngine: PyMuPDF failed on '%s': %s — using text mode.",
                pdf_path.name,
                exc,
            )
            return cls([], source="pdf_error")

        return cls(blocks, source="pdf")

    @classmethod
    def from_text(cls, text: str) -> "LayoutEngine":
        """Build from plain text using heuristic font-size and bold inference."""
        blocks: list[LayoutBlock] = []
        y = 0.0
        for idx, raw_line in enumerate(text.splitlines()):
            line = raw_line.rstrip()
            indent = len(line) - len(line.lstrip())
            blocks.append(LayoutBlock(
                text=line,
                page_num=0,
                x0=float(indent),
                y0=y,
                x1=float(max(len(line), 1)),
                y1=y + 14.0,
                font_size=_text_infer_size(line),
                is_bold=_text_infer_bold(line),
                is_italic=False,
                block_index=idx,
            ))
            y += 14.0
        return cls(blocks, source="text")

    # ── public properties ─────────────────────────────────────────────────── #

    @property
    def blocks(self) -> list[LayoutBlock]:
        return self._blocks

    @property
    def non_empty_blocks(self) -> list[LayoutBlock]:
        return [b for b in self._blocks if not b.is_empty]

    def __iter__(self) -> Iterator[LayoutBlock]:
        return iter(self._blocks)

    def __len__(self) -> int:
        return len(self._blocks)

    # ── heading detection ────────────────────────────────────────────────────

    def is_heading(self, block: LayoutBlock) -> bool:
        """Return True if *block* looks like a section heading.

        PDF mode:  font_size ≥ threshold  OR  (bold AND size ≥ avg)  OR  ALL-CAPS.
        Text mode: heuristic ALL-CAPS / Title-Case check.
        """
        if block.is_empty:
            return False
        if self.source == "pdf":
            size_head = block.font_size >= self._heading_threshold
            bold_head = block.is_bold and block.font_size >= self._avg_font_size
            caps_head = _text_looks_like_heading(block.stripped)
            return size_head or bold_head or caps_head
        return _text_looks_like_heading(block.stripped)

    # ── utilities ─────────────────────────────────────────────────────────── #

    def plain_text(self) -> str:
        """Return the concatenated text of all blocks, preserving line order."""
        return "\n".join(b.text for b in self._blocks)

    # ── internals ─────────────────────────────────────────────────────────── #

    @staticmethod
    def _find_pdf(workspace_dir: Path) -> Path | None:
        """Look for a file named original.pdf inside the workspace."""
        if not workspace_dir or not workspace_dir.is_dir():
            return None
        for suffix in _PDF_SUFFIXES:
            candidate = workspace_dir / f"original{suffix}"
            if candidate.is_file():
                return candidate
        # Fallback: any PDF in the workspace.
        for f in workspace_dir.iterdir():
            if f.suffix.lower() in _PDF_SUFFIXES and f.is_file():
                return f
        return None

    def _calc_avg(self) -> float:
        sizes = [b.font_size for b in self._blocks if not b.is_empty]
        return sum(sizes) / len(sizes) if sizes else 12.0
