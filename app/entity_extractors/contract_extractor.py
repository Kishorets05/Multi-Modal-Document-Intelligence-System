"""Contract entity extractor — Hybrid Intelligent pipeline.

Pipeline
--------
1. LayoutEngine  (PDF real layout OR text heuristics)
2. SectionParser (known contract headings)
3. spaCy NLP     (ORG for parties, DATE for dates, GPE for jurisdiction)
4. Rule Engine   (clause-scoped regex patterns, cascade fallback)
5. ValidationEngine (confidence wrapping)
"""
from __future__ import annotations

import re
from typing import Final

from app.entity_extractors.base import BaseEntityExtractor
from app.entity_extractors.validation_engine import (
    make_field,
    pdf_bonus,
    validate_date_spacy,
    validate_org,
)

# ─────────────────────────────────────────────────────────────────────────── #
#  Regex — dates only                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

_DATE_RE: Final = re.compile(
    r"(?:"
    r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}"
    r"|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}"
    r")",
    re.IGNORECASE,
)


def _extract_date(text: str) -> str:
    m = _DATE_RE.search(text)
    return m.group().strip() if m else ""


def _all_dates(text: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for m in _DATE_RE.finditer(text):
        d = m.group().strip()
        if d not in seen:
            seen.add(d)
            result.append(d)
    return result


# ─────────────────────────────────────────────────────────────────────────── #
#  Contract section headings                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

_CONTRACT_SECTIONS: Final[set[str]] = {
    "parties", "party", "recitals", "recital", "preamble", "whereas",
    "background", "definitions", "definition",
    "scope", "scope of work", "scope of services",
    "term", "term and termination", "termination", "termination and expiry",
    "effective date", "commencement",
    "payment terms", "payment schedule", "payments",
    "governing law", "jurisdiction", "choice of law",
    "confidentiality", "non-disclosure", "non disclosure",
    "intellectual property",
    "liability", "limitation of liability",
    "indemnification", "indemnify",
    "warranties", "representations",
    "dispute resolution", "arbitration",
    "force majeure",
    "entire agreement",
    "amendments", "modifications",
    "notices", "notice",
    "signatures", "execution", "annexures", "schedules", "exhibits",
}

# ─────────────────────────────────────────────────────────────────────────── #
#  Party extraction helpers (cascade)                                         #
# ─────────────────────────────────────────────────────────────────────────── #

def _parties_from_between(text: str) -> tuple[str, str]:
    m = re.search(
        r"between\s+"
        r"([A-Z][^\n,]{2,80}?)"
        r"\s+(?:and|&)\s+"
        r"([A-Z][^\n,]{2,80}?)"
        r"(?:\s*[,\(\n]|hereinafter|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip().strip("("), m.group(2).strip().strip("(")
    return "", ""


def _parties_from_hereinafter(text: str) -> list[str]:
    result: list[str] = []
    for m in re.finditer(
        r'([A-Z][^\n,]{2,80}?)\s*[,\(]?\s*hereinafter\s+'
        r'(?:referred\s+to\s+as\s+)?["\']?[A-Za-z\s]+["\']?',
        text,
        re.IGNORECASE,
    ):
        p = m.group(1).strip()
        if p:
            result.append(p)
    return result


def _parties_from_labels(text: str) -> list[str]:
    result: list[str] = []
    for m in re.finditer(
        r"(?:party\s*[\"']?(?:a|b|1|2|one|two)[\"']?|"
        r"first\s+party|second\s+party|"
        r"client|service\s+provider|vendor|contractor|employer|employee)"
        r"\s*[:\-]\s*([^\n,;]{2,80})",
        text,
        re.IGNORECASE,
    ):
        p = m.group(1).strip()
        if p:
            result.append(p)
    return result


# ─────────────────────────────────────────────────────────────────────────── #
#  Clause-scoped patterns                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

_EFF_DATE_CTX_RE: Final = re.compile(
    r"(?:effective(?:\s+as\s+of)?|entered\s+into(?:\s+as\s+of)?|"
    r"made\s+(?:and\s+)?(?:entered\s+)?as\s+of|"
    r"dated?(?:\s+this)?|commenc(?:es?|ing)(?:\s+on)?)"
    r"[^.]{0,80}",
    re.IGNORECASE,
)
_TERM_DATE_CTX_RE: Final = re.compile(
    r"(?:terminat(?:es?|ion)|expir(?:es?|ation|y)|end(?:s)?\s+on|"
    r"shall\s+(?:expire|end|terminate))"
    r"[^.]{0,80}",
    re.IGNORECASE,
)
_GOV_LAW_RE: Final = re.compile(
    r"(?:governed\s+by(?:\s+the\s+laws?\s+of)?|"
    r"governing\s+law\s*[:\-]?\s*|"
    r"jurisdiction\s*[:\-]?\s*|"
    r"subject\s+to\s+the\s+laws?\s+of)"
    r"\s*([^\n.;,]{3,80})",
    re.IGNORECASE,
)
_PAY_TERMS_RE: Final = re.compile(
    r"(?:payment\s+terms?\s*[:\-]?\s*|"
    r"payment\s+(?:shall\s+be\s+)?(?:due|made)\s*(?:within\s*)?)"
    r"([^\n.;]{5,120})",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────── #
#  Main extractor                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class ContractEntityExtractor(BaseEntityExtractor):
    """Extract structured entities from a contract.

    Priority chain per field
    ------------------------
    party_one/two    : spaCy ORG on preamble → between/hereinafter → labels
    effective_date   : spaCy DATE on context → regex context → preamble dates
    termination_date : spaCy DATE on termination section → regex
    governing_law    : spaCy GPE on governing-law section → regex
    payment_terms    : regex (no suitable NER label)
    """

    _HEADING_NAMES: set[str] = _CONTRACT_SECTIONS

    def extract(self, text: str) -> dict:
        is_pdf = self._is_pdf_layout()
        layout_bonus = pdf_bonus(is_pdf)
        nlp = self._nlp()

        # ── spaCy pass ────────────────────────────────────────────────────
        doc = self._doc(text)
        spacy_orgs  = self._ents(doc, "ORG")
        spacy_dates = self._ents(doc, "DATE")
        spacy_gpe   = self._ents(doc, "GPE")

        # ── Layout + section parser ───────────────────────────────────────
        parser = self._build_parser(text)
        preamble_blocks = parser.get_preamble_blocks()
        preamble = "\n".join(b.text for b in preamble_blocks)
        search_scope = preamble if preamble.strip() else text

        # ── Parties ───────────────────────────────────────────────────────
        party_one = ""
        party_two = ""
        party_conf = 0.0

        # Strategy 1: spaCy ORG on preamble (highest confidence)
        if spacy_orgs:
            party_one  = spacy_orgs[0]
            party_two  = spacy_orgs[1] if len(spacy_orgs) > 1 else ""
            party_conf = 0.88 + layout_bonus

        # Strategy 2: "between X and Y" in preamble
        if not party_one:
            p1, p2 = _parties_from_between(search_scope)
            if p1:
                party_one  = p1
                party_two  = p2
                party_conf = 0.80 + layout_bonus

        # Strategy 3: hereinafter patterns
        if not party_one:
            hafter = _parties_from_hereinafter(text)
            if hafter:
                party_one  = hafter[0]
                party_two  = hafter[1] if len(hafter) > 1 else ""
                party_conf = 0.72 + layout_bonus

        # Strategy 4: labeled party lines
        if not party_one:
            labeled = _parties_from_labels(text)
            if labeled:
                party_one  = labeled[0]
                party_two  = labeled[1] if len(labeled) > 1 else ""
                party_conf = 0.65 + layout_bonus

        # Clean org names with validator
        if party_one:
            v, c = validate_org(party_one, spacy_orgs)
            if v:
                party_one  = v
                party_conf = max(party_conf, c + layout_bonus)
        if party_two:
            v2, _ = validate_org(party_two, spacy_orgs)
            if v2:
                party_two = v2

        # ── Effective date ────────────────────────────────────────────────
        eff_date  = ""
        eff_conf  = 0.0

        eff_m = _EFF_DATE_CTX_RE.search(text)
        if eff_m:
            eff_date = _extract_date(eff_m.group())
            if eff_date:
                eff_conf = 0.78 + layout_bonus

        if not eff_date and spacy_dates:
            eff_date = spacy_dates[0]
            eff_conf = 0.88 + layout_bonus

        if not eff_date:
            dates = _all_dates(preamble)
            if dates:
                eff_date = dates[0]
                eff_conf = 0.72 + layout_bonus

        # Validate with spaCy
        if eff_date:
            dv, dc = validate_date_spacy(eff_date, nlp)
            if dv:
                eff_date = dv
                eff_conf = max(eff_conf, dc + layout_bonus)

        # ── Termination date ──────────────────────────────────────────────
        term_date = ""
        term_conf = 0.0

        term_text = parser.get_section_text("termination", "term")
        term_scope = term_text if term_text.strip() else text
        term_m = _TERM_DATE_CTX_RE.search(term_scope)
        if term_m:
            term_date = _extract_date(term_m.group())
            if term_date:
                term_conf = 0.78 + layout_bonus

        if not term_date and term_text.strip():
            tdates = _all_dates(term_text)
            if tdates:
                term_date = tdates[0]
                term_conf = 0.72 + layout_bonus

        if term_date:
            dv, dc = validate_date_spacy(term_date, nlp)
            if dv:
                term_date = dv
                term_conf = max(term_conf, dc + layout_bonus)

        # ── Governing law ─────────────────────────────────────────────────
        gov_law  = ""
        gov_conf = 0.0

        gov_text = parser.get_section_text("governing law", "jurisdiction")
        # spaCy GPE on governing law section or full text
        gpe_scope = gov_text if gov_text.strip() else ""
        if gpe_scope and spacy_gpe:
            gov_law  = spacy_gpe[0]
            gov_conf = 0.85 + layout_bonus
        else:
            gov_m = _GOV_LAW_RE.search(gov_text or text)
            if gov_m:
                gov_law  = gov_m.group(1).strip().rstrip(".,;")
                gov_conf = 0.70 + layout_bonus
            elif spacy_gpe:
                gov_law  = spacy_gpe[0]
                gov_conf = 0.75 + layout_bonus

        # ── Payment terms ─────────────────────────────────────────────────
        pay_terms = ""
        pay_conf  = 0.0

        pay_text = parser.get_section_text("payment terms", "payments")
        pay_m = _PAY_TERMS_RE.search(pay_text or text)
        if pay_m:
            pay_terms = pay_m.group(1).strip().rstrip(".,;")
            pay_conf  = 0.72 + layout_bonus

        # party_two confidence must be 0.0 when no value was found; using the
        # same high confidence as party_one for an empty field is misleading.
        party_two_conf = round(min(party_conf, 1.0), 3) if party_two else 0.0

        return {
            "party_one":        make_field(party_one,  round(min(party_conf, 1.0), 3)),
            "party_two":        make_field(party_two,  party_two_conf),
            "effective_date":   make_field(eff_date,   round(min(eff_conf,  1.0), 3)),
            "termination_date": make_field(term_date,  round(min(term_conf, 1.0), 3)),
            "governing_law":    make_field(gov_law,    round(min(gov_conf,  1.0), 3)),
            "payment_terms":    make_field(pay_terms,  round(min(pay_conf,  1.0), 3)),
        }
