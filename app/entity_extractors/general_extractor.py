"""General document entity extractor — Hybrid Intelligent pipeline.

Pipeline
--------
1. spaCy NLP  (PERSON and ORG entities — new fields)
2. Layout     (largest-font block for title)
3. Regex      (email, phone, URL — validation only)
4. ValidationEngine (confidence wrapping)
"""
from __future__ import annotations

import re
from typing import Final

from app.entity_extractors.base import BaseEntityExtractor
from app.entity_extractors.validation_engine import (
    make_field,
    make_list_field,
    pdf_bonus,
    validate_email,
    validate_phone,
)

# ─────────────────────────────────────────────────────────────────────────── #
#  Regex — contact info and URLs                                              #
# ─────────────────────────────────────────────────────────────────────────── #

_EMAIL_RE: Final = re.compile(
    r"(?<![a-zA-Z0-9._%+\-])"
    r"[a-zA-Z0-9._%+\-]{1,64}"
    r"@"
    r"[a-zA-Z0-9.\-]{1,253}"
    r"\.[a-zA-Z]{2,}"
    r"(?![a-zA-Z0-9._%+\-])"
)
_PHONE_RE: Final = re.compile(
    r"(?<!\d)"
    r"(?:\+\d{1,3}[\s\-.]?)?"
    r"(?:\(?\d{2,5}\)?[\s\-.]?)?"
    r"\d{3,5}"
    r"[\s\-.]"                # explicit separator required
    r"\d{4,5}"
    r"(?!\d)"
)
_URL_RE: Final = re.compile(
    r"https?://[^\s<>\"'\]\[)]+|"
    r"www\.[a-zA-Z0-9\-]{2,}\.[a-zA-Z]{2,}[^\s<>\"'\]\[)]*"
)

# Title detection heuristics
_MAX_TITLE_LEN: Final[int] = 120
_MIN_TITLE_LEN: Final[int] = 2
_SECTION_LABEL_WORDS: Final[frozenset[str]] = frozenset({
    "introduction", "conclusion", "references", "abstract", "summary",
    "contents", "table of contents", "index", "appendix", "glossary",
    "acknowledgements", "acknowledgments", "background", "scope",
    "overview", "objectives", "methodology", "results", "discussion",
})


def _is_candidate_title(line: str) -> bool:
    s = line.strip()
    if not s or len(s) < _MIN_TITLE_LEN or len(s) > _MAX_TITLE_LEN:
        return False
    if _EMAIL_RE.search(s) or _URL_RE.search(s):
        return False
    # Exclude pure section-label lines (e.g. "Introduction", "References")
    # so that they are never mistakenly returned as the document title.
    if s.lower().rstrip(":.") in _SECTION_LABEL_WORDS:
        return False
    return True


def _prefer_heading(line: str) -> bool:
    s = line.strip()
    return bool(s) and (s.isupper() or s.istitle())


# ─────────────────────────────────────────────────────────────────────────── #
#  Main extractor                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class GeneralEntityExtractor(BaseEntityExtractor):
    """Extract generic entities from any unclassified document.

    Fields
    ------
    title          : layout largest-font block (PDF) or first heading line
    emails[]       : regex + validate_email
    phones[]       : regex + validate_phone
    urls[]         : regex
    persons[]      : spaCy PERSON  (NEW)
    organizations[]: spaCy ORG     (NEW)
    """

    def extract(self, text: str) -> dict:
        is_pdf = self._is_pdf_layout()
        layout_bonus = pdf_bonus(is_pdf)

        # ── spaCy pass ────────────────────────────────────────────────────
        doc = self._doc(text)
        spacy_persons = self._ents(doc, "PERSON")
        spacy_orgs    = self._ents(doc, "ORG")

        # ── Title ─────────────────────────────────────────────────────────
        title_val  = ""
        title_conf = 0.0

        # PDF mode: use layout largest-font block.
        if is_pdf and self._workspace_dir:
            try:
                from app.entity_extractors.layout_engine import LayoutEngine
                engine = LayoutEngine.from_workspace(self._workspace_dir, text)
                non_empty = engine.non_empty_blocks
                if non_empty:
                    largest = max(non_empty, key=lambda b: b.font_size)
                    if _is_candidate_title(largest.stripped):
                        title_val  = largest.stripped
                        title_conf = 0.85 + layout_bonus
            except Exception:
                pass

        # Fallback: first heading-like line.
        if not title_val:
            lines = [ln.strip() for ln in text.splitlines()]
            first_candidate = ""
            first_heading   = ""
            for line in lines:
                if not line:
                    continue
                if not _is_candidate_title(line):
                    continue
                if not first_candidate:
                    first_candidate = line
                if not first_heading and _prefer_heading(line):
                    first_heading = line
                    break
            title_val  = first_heading or first_candidate
            title_conf = 0.75 if title_val else 0.0

        # ── Emails ────────────────────────────────────────────────────────
        emails: list[dict] = []
        seen_emails: set[str] = set()
        for m in _EMAIL_RE.finditer(text):
            raw = m.group()
            val, conf = validate_email(raw)
            if val and val not in seen_emails:
                seen_emails.add(val)
                emails.append({"value": val, "confidence": conf})

        # ── Phones ────────────────────────────────────────────────────────
        phones: list[dict] = []
        seen_phones: set[str] = set()
        for m in _PHONE_RE.finditer(text):
            raw = m.group().strip()
            val, conf = validate_phone(raw)
            if val and val not in seen_phones:
                seen_phones.add(val)
                phones.append({"value": val, "confidence": conf})

        # ── URLs ──────────────────────────────────────────────────────────
        urls: list[dict] = []
        seen_urls: set[str] = set()
        for m in _URL_RE.finditer(text):
            url = m.group().rstrip(".,;:)'\"")
            if url not in seen_urls:
                seen_urls.add(url)
                urls.append({"value": url, "confidence": 0.85})

        # ── Persons (spaCy PERSON) ─────────────────────────────────────── #
        person_conf = round(0.88 + layout_bonus, 3)
        persons = [{"value": p, "confidence": person_conf} for p in spacy_persons]

        # ── Organisations (spaCy ORG) ──────────────────────────────────── #
        org_conf = round(0.85 + layout_bonus, 3)
        organizations = [{"value": o, "confidence": org_conf} for o in spacy_orgs]

        return {
            "title":         make_field(title_val, round(min(title_conf, 1.0), 3)),
            "emails":        emails,
            "phones":        phones,
            "urls":          urls,
            "persons":       persons,
            "organizations": organizations,
        }
