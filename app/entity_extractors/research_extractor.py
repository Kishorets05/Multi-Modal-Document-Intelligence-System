"""Research paper entity extractor — Hybrid Intelligent pipeline.

Pipeline
--------
1. State machine (HEADER → ABSTRACT → BODY → REFS) — unchanged structure
2. spaCy NLP    (PERSON for authors)
3. Layout       (largest-font block for title validation)
4. Regex        (reference entry counting, keyword splitting — unchanged)
5. ValidationEngine (confidence wrapping)
"""
from __future__ import annotations

import re
from typing import Final

from app.entity_extractors.base import BaseEntityExtractor
from app.entity_extractors.validation_engine import (
    make_field,
    make_list_field,
    pdf_bonus,
)

# ─────────────────────────────────────────────────────────────────────────── #
#  Section boundary patterns (state machine)                                  #
# ─────────────────────────────────────────────────────────────────────────── #

_ABSTRACT_RE: Final = re.compile(r"^\s*abstract\s*$", re.IGNORECASE)
_KEYWORDS_RE: Final = re.compile(
    r"^\s*(?:keywords?|index\s+terms?|key\s+words?)\s*[:\-]?\s*(.*)",
    re.IGNORECASE,
)
_INTRO_RE: Final = re.compile(
    r"^\s*(?:\d+[\.\ )]\s+)?(?:i\.?\s+)?introduction\s*[:\.]?\s*$", re.IGNORECASE
)
_SECTION_NUM_RE: Final = re.compile(
    r"^\s*(?:[IVXivx]+[\.\ )]\s+|\d+[\.\ )]\s+|\d+\.\d+\.?\s+)\S"
)
_REFS_RE: Final = re.compile(
    r"^\s*(?:\d+[\.\ )\ ]?\ +)?(?:references?|bibliography)\s*[:\.]?\s*$",
    re.IGNORECASE,
)
_ACK_RE: Final = re.compile(
    r"^\s*(?:\d+[\.\ )\ ]?\ +)?(?:acknowledgements?|acknowledgments?)\s*[:\.]?\s*$",
    re.IGNORECASE,
)
_REF_ENTRY_RE: Final = re.compile(r"^\s*(?:\[\d+\]|\d+[\.\ )])\s+\S")

# ─────────────────────────────────────────────────────────────────────────── #
#  Author detection helpers                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

_AFFILIATION_RE: Final = re.compile(
    r"(?:university|institute|college|department|dept\.|school|laboratory|"
    r"lab\b|corp\.|inc\.|ltd\.?|llc|pvt\.?|@|\.com|\.edu|\.org|"
    r"address|email|corresponding|orcid|p\.?\s*o\.?\s+box|"
    r"abstract|introduction|keywords?)",
    re.IGNORECASE,
)
_AUTHOR_LINE_RE: Final = re.compile(
    r"^[A-Z][a-zA-Z\.\-]+"
    r"(?:\s+[A-Z][a-zA-Z\.\-]+)+"
    r"(?:[,*†‡§\s\d]+"
    r"[A-Z][a-zA-Z\.\-]+"
    r"(?:\s+[A-Z][a-zA-Z\.\-]+)*)*"
    r"\s*[*†‡§\d]?\s*$"
)


def _split_authors(raw: str) -> list[str]:
    parts = re.split(r",\s*|\s+and\s+|\s*&\s*", raw)
    cleaned: list[str] = []
    for part in parts:
        name = re.sub(r"[\d*†‡§¶\s]+$", "", part).strip(" .,")
        if name and len(name) >= 2 and name[0].isupper():
            cleaned.append(name)
    return cleaned


# ─────────────────────────────────────────────────────────────────────────── #
#  Main extractor                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class ResearchEntityExtractor(BaseEntityExtractor):
    """Extract structured entities from a research paper.

    Priority chain per field
    ------------------------
    title   : layout largest-font block in HEADER state (with PDF bonus)
    authors : spaCy PERSON on header text → author-line heuristic
    abstract: state-machine ABSTRACT section (structural → confidence 0.90)
    keywords: keywords line/inline (structural → confidence 0.88)
    references_count: regex counting (deterministic → confidence 0.95)
    """

    def extract(self, text: str) -> dict:
        is_pdf = self._is_pdf_layout()
        layout_bonus = pdf_bonus(is_pdf)

        lines = [ln.rstrip() for ln in text.splitlines()]
        n = len(lines)

        # ── State machine pass ────────────────────────────────────────────
        state = "HEADER"
        abstract_lines: list[str] = []
        header_lines: list[str] = []
        keywords_raw = ""
        ref_count = 0

        i = 0
        while i < n:
            line   = lines[i]
            stripped = line.strip()

            if _REFS_RE.match(stripped):
                state = "REFS"
                i += 1
                continue

            if state == "REFS":
                if _REF_ENTRY_RE.match(stripped):
                    ref_count += 1
                i += 1
                continue

            kw_m = _KEYWORDS_RE.match(stripped)
            if kw_m:
                state = "BODY"
                inline = kw_m.group(1).strip()
                if inline:
                    keywords_raw = inline
                else:
                    j = i + 1
                    while j < n and not lines[j].strip():
                        j += 1
                    if j < n:
                        keywords_raw = lines[j].strip()
                        i = j
                i += 1
                continue

            if _ABSTRACT_RE.match(stripped):
                state = "ABSTRACT"
                i += 1
                continue

            if state == "ABSTRACT":
                if stripped and (
                    _INTRO_RE.match(stripped)
                    or _SECTION_NUM_RE.match(stripped)
                    or _ACK_RE.match(stripped)
                    or _REFS_RE.match(stripped)
                ):
                    state = "BODY"
                    continue
                if stripped:
                    abstract_lines.append(stripped)
                i += 1
                continue

            if _ACK_RE.match(stripped) or (
                state == "BODY" and _SECTION_NUM_RE.match(stripped)
            ):
                i += 1
                continue

            if state == "HEADER" and stripped:
                if not _AFFILIATION_RE.search(stripped):
                    header_lines.append(stripped)

            i += 1

        # ── Extract author candidates and title candidates ─────────────────
        author_lines: list[str] = []
        title_candidates: list[str] = []

        for hl in header_lines:
            auth_label = re.match(r"^authors?\s*[:\-]\s*(.*)", hl, re.IGNORECASE)
            if auth_label:
                raw = auth_label.group(1).strip()
                if raw:
                    author_lines.append(raw)
            elif _AUTHOR_LINE_RE.match(hl):
                author_lines.append(hl)
            else:
                title_candidates.append(hl)

        # ── Title: prefer largest-font layout block ───────────────────────
        title_val  = ""
        title_conf = 0.0

        if is_pdf and self._workspace_dir:
            try:
                from app.entity_extractors.layout_engine import LayoutEngine
                engine = LayoutEngine.from_workspace(self._workspace_dir, text)
                # Restrict to HEADER blocks (those before the abstract).
                header_text_set = set(header_lines)
                header_blocks = [
                    b for b in engine.non_empty_blocks
                    if b.stripped in header_text_set
                    and not _AFFILIATION_RE.search(b.stripped)
                ]
                if header_blocks:
                    largest = max(header_blocks, key=lambda b: b.font_size)
                    title_val  = largest.stripped
                    title_conf = 0.90 + layout_bonus
            except Exception:
                pass

        if not title_val and title_candidates:
            title_val  = max(title_candidates, key=len)
            title_conf = 0.75 + layout_bonus

        # ── Authors: spaCy PERSON on header text ──────────────────────────
        header_text_str = "\n".join(header_lines)
        doc = self._doc(header_text_str) if header_text_str.strip() else None
        spacy_persons_header = self._ents(doc, "PERSON")

        authors_list: list[str] = []
        authors_conf = 0.0
        seen_authors: set[str] = set()

        if spacy_persons_header:
            for name in spacy_persons_header:
                if name not in seen_authors:
                    seen_authors.add(name)
                    authors_list.append(name)
            authors_conf = 0.88 + layout_bonus
        else:
            # Fallback: regex author-line heuristic
            for al in author_lines:
                for name in _split_authors(al):
                    if name not in seen_authors:
                        seen_authors.add(name)
                        authors_list.append(name)
            authors_conf = 0.70 + layout_bonus if authors_list else 0.0

        # ── Abstract ──────────────────────────────────────────────────────
        abstract_val  = " ".join(abstract_lines)
        abstract_conf = round(0.90 + layout_bonus, 3) if abstract_val else 0.0

        # ── Keywords ──────────────────────────────────────────────────────
        keywords_list: list[str] = []
        if keywords_raw:
            kws = re.split(r"[;,|·•]", keywords_raw)
            keywords_list = [k.strip(" .-·*†") for k in kws if k.strip(" .-·*†")]
        kw_conf = round(0.88 + layout_bonus, 3) if keywords_list else 0.0

        # ── References count ──────────────────────────────────────────────
        ref_conf = 0.95 if ref_count > 0 else 0.0

        return {
            "title":            make_field(title_val, round(min(title_conf, 1.0), 3)),
            "authors":          [
                {"value": a, "confidence": round(min(authors_conf, 1.0), 3)}
                for a in authors_list
            ],
            "abstract":         make_field(abstract_val, abstract_conf),
            "keywords":         make_list_field(keywords_list, kw_conf),
            "references_count": {"value": ref_count, "confidence": ref_conf},
        }
