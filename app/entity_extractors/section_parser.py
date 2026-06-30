"""Section parser — converts LayoutEngine output into named document sections.

Public API::

    from app.entity_extractors.section_parser import SectionParser, Section

    parser = SectionParser(engine, heading_names={"experience", "education", ...})

    # Get a dict of all sections:
    sections = parser.split_into_sections()

    # Get the text of a specific section:
    text = parser.get_section_text("experience")

    # Get content between two headings:
    blocks = parser.extract_between("experience", stop_before=["projects"])

    # Find a heading block:
    head = parser.find_heading("technical skills")

    # Find the next heading after a given block:
    next_head = parser.find_next_heading(head)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.entity_extractors.layout_engine import LayoutBlock, LayoutEngine


# ─────────────────────────────────────────────────────────────────────────── #
#  Section data model                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class Section:
    """A named section: its heading block plus all content blocks."""

    name: str                  # normalised heading text (lowercase, no punct)
    heading_block: LayoutBlock
    content_blocks: list[LayoutBlock] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Content as a single multi-line string."""
        return "\n".join(b.text for b in self.content_blocks if b.text.strip())

    @property
    def lines(self) -> list[str]:
        """Content as a list of non-empty strings."""
        return [b.text for b in self.content_blocks if b.text.strip()]

    @property
    def stripped_lines(self) -> list[str]:
        """Content lines with leading/trailing whitespace removed."""
        return [b.stripped for b in self.content_blocks if b.stripped]

    def __repr__(self) -> str:  # pragma: no cover
        return f"Section(name={self.name!r}, blocks={len(self.content_blocks)})"


# ─────────────────────────────────────────────────────────────────────────── #
#  Heading normalisation                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def _norm(text: str) -> str:
    """Lowercase, collapse whitespace, strip trailing colon/period."""
    s = re.sub(r"\s+", " ", text.strip())
    return s.rstrip(":.").strip().lower()


# ─────────────────────────────────────────────────────────────────────────── #
#  Section Parser                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

class SectionParser:
    """Splits a document into named sections and provides extraction helpers.

    Parameters
    ----------
    engine :
        A ``LayoutEngine`` instance produced by ``LayoutEngine.from_pdf()`` or
        ``LayoutEngine.from_text()``.
    heading_names :
        Optional explicit set of known section headings (lowercase).
        When provided, a block is treated as a **section boundary** ONLY if
        its normalised text is a member of this set.  This prevents false
        positives such as ALL-CAPS job titles (e.g. "CLOUD COMPUTING INTERN")
        from splitting a section prematurely.
        When ``None``, every block the ``LayoutEngine`` identifies as a heading
        is used as a boundary.
    """

    def __init__(
        self,
        engine: LayoutEngine,
        heading_names: set[str] | None = None,
    ) -> None:
        self._engine = engine
        # Normalise the caller-supplied heading names once.
        self._heading_names: set[str] | None = (
            {_norm(h) for h in heading_names} if heading_names is not None else None
        )
        self._sections: dict[str, Section] | None = None   # computed lazily

    # ── core helpers (required by spec) ─────────────────────────────────── #

    def split_into_sections(self) -> dict[str, Section]:
        """Return ``{normalised_heading: Section}`` for the entire document.

        Sections are accumulated in reading order.  Content ends at the
        next recognised heading.  Blocks that appear before the first
        heading are accessible via ``get_preamble_blocks()``.
        """
        if self._sections is not None:
            return self._sections

        result: dict[str, Section] = {}
        current: Section | None = None

        for block in self._engine.blocks:
            if not block.text.strip():
                continue
            if self._is_section_boundary(block):
                name = _norm(block.stripped)
                section = Section(name=name, heading_block=block)
                # If the same heading appears more than once, keep the first.
                result.setdefault(name, section)
                current = section
            elif current is not None:
                current.content_blocks.append(block)

        self._sections = result
        return result

    def find_heading(self, *names: str) -> LayoutBlock | None:
        """Return the heading block for the first matching section name.

        Matching is case-insensitive and tolerates prefix / substring matches
        (e.g. ``"experience"`` matches ``"work experience"``).
        Returns ``None`` when no match exists.
        """
        sections = self.split_into_sections()
        for name in names:
            norm = _norm(name)
            # 1. Exact match.
            if norm in sections:
                return sections[norm].heading_block
            # 2. Substring match.
            for key, sec in sections.items():
                if norm in key or key in norm:
                    return sec.heading_block
        return None

    def find_next_heading(self, after: LayoutBlock) -> LayoutBlock | None:
        """Return the next section-boundary block that comes after *after*."""
        past = False
        for block in self._engine.blocks:
            if block.block_index == after.block_index:
                past = True
                continue
            if past and self._is_section_boundary(block):
                return block
        return None

    def extract_between(
        self,
        start_name: str,
        stop_before: list[str] | None = None,  # retained for API/call-site compat
    ) -> list[LayoutBlock]:
        """Return the content blocks that belong to *start_name*.

        Collection begins immediately after the *start_name* heading and
        stops at the **next recognised section boundary**.

        The *stop_before* parameter is retained for backward compatibility and
        call-site documentation, but the effective stop condition is always
        the next boundary recognised by ``_is_section_boundary`` — every path
        through that check results in a break, so being selective about which
        boundary name to stop on is redundant and risks cross-section bleed.

        Parameters
        ----------
        start_name :
            The name of the section to extract (case-insensitive).
        stop_before :
            Retained for API compatibility; does not alter the stop condition.
        """
        start_head = self.find_heading(start_name)
        if start_head is None:
            return []

        result: list[LayoutBlock] = []
        collecting = False

        for block in self._engine.blocks:
            if block.block_index == start_head.block_index:
                collecting = True
                continue
            if not collecting:
                continue
            if self._is_section_boundary(block):
                # Stop at every recognised boundary — the primary guard against
                # cross-section bleed, regardless of which heading comes next.
                break
            result.append(block)

        return result

    # ── convenience helpers ──────────────────────────────────────────────── #

    def get_section_text(self, *section_names: str) -> str:
        """Return joined content text of the first matching section."""
        sections = self.split_into_sections()
        for name in section_names:
            norm = _norm(name)
            if norm in sections:
                return sections[norm].text
            for key, sec in sections.items():
                if norm in key or key in norm:
                    return sec.text
        return ""

    def get_section_lines(self, *section_names: str) -> list[str]:
        """Return stripped content lines of the first matching section."""
        sections = self.split_into_sections()
        for name in section_names:
            norm = _norm(name)
            if norm in sections:
                return sections[norm].stripped_lines
            for key, sec in sections.items():
                if norm in key or key in norm:
                    return sec.stripped_lines
        return []

    def get_preamble_blocks(self) -> list[LayoutBlock]:
        """Return blocks that appear before the first section heading."""
        first_head_idx: int | None = None
        for block in self._engine.blocks:
            if self._is_section_boundary(block):
                first_head_idx = block.block_index
                break

        if first_head_idx is None:
            return list(self._engine.non_empty_blocks)

        return [
            b for b in self._engine.blocks
            if b.block_index < first_head_idx and b.stripped
        ]

    def get_top_blocks(self, n: int = 10) -> list[LayoutBlock]:
        """Return the first *n* non-empty blocks regardless of headings."""
        return [b for b in self._engine.non_empty_blocks][:n]

    @property
    def all_blocks(self) -> list[LayoutBlock]:
        """All non-empty layout blocks in reading order.

        Use this instead of accessing ``_engine`` directly so that callers are
        decoupled from the internal ``LayoutEngine`` implementation.
        """
        return self._engine.non_empty_blocks

    # ── internals ────────────────────────────────────────────────────────── #

    def _is_section_boundary(self, block: LayoutBlock) -> bool:
        """A block is a section boundary if the layout engine marks it as a
        heading AND (when ``heading_names`` is provided) its normalised text
        is a member of the known-headings set."""
        if not self._engine.is_heading(block):
            return False
        if self._heading_names is not None:
            return _norm(block.stripped) in self._heading_names
        return True
