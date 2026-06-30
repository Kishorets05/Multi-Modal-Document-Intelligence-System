"""Validation Engine — field validators with confidence scoring.

Every public function returns a ``(cleaned_value, confidence)`` tuple.
Rules:
- If validation fails the value is ``""`` (or ``[]``) and confidence is ``0.0``.
- Never raise — absorb errors and return empty/0.0.
- Confidence is additive: base score + source bonuses, capped at 1.0.

Confidence base values
----------------------
spaCy NER entity match      → 0.90
EntityRuler pattern match   → 0.85
Regex-only extraction       → 0.72
Heuristic (layout/string)   → 0.60

Confidence bonuses (additive, capped at 1.0)
--------------------------------------------
+ 0.05  PDF layout source (real font sizes available)
+ 0.03  Regex validation passes on an NER-extracted value
"""
from __future__ import annotations

import re
from typing import Final

# ─────────────────────────────────────────────────────────────────────────── #
#  Compiled patterns used ONLY for validation (not for extraction)            #
# ─────────────────────────────────────────────────────────────────────────── #

_EMAIL_VAL_RE: Final = re.compile(
    r"^[a-zA-Z0-9._%+\-]{1,64}"
    r"@"
    r"[a-zA-Z0-9.\-]{1,253}"
    r"\.[a-zA-Z]{2,}$"
)

_PHONE_VAL_RE: Final = re.compile(
    r"(?<!\d)"
    r"(?:\+\d{1,3}[\s\-.]?)?"          # optional country code
    r"(?:\(?\d{2,5}\)?[\s\-.]?)?"      # optional area code
    r"\d{3,5}"
    r"[\s\-.]"                          # explicit separator required
    r"\d{4,5}"
    r"(?!\d)"
)

_MONEY_VAL_RE: Final = re.compile(
    r"[₹$€£¥]?\s*[\d,]+\.?\d*"
)

_DATE_VAL_RE: Final = re.compile(
    r"(?:"
    r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}"
    r"|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}"
    r")",
    re.IGNORECASE,
)

# Job-title keywords: lines containing these are unlikely to be names.
_TITLE_WORDS: Final[frozenset[str]] = frozenset({
    "developer", "engineer", "designer", "analyst", "manager",
    "architect", "consultant", "intern", "specialist", "officer",
    "coordinator", "director", "head", "lead", "senior", "junior",
    "associate", "executive", "president", "ceo", "cto", "cfo",
    "frontend", "backend", "fullstack", "full-stack", "devops",
})

# Noise words that are never valid organisation names.
_ORG_NOISE: Final[frozenset[str]] = frozenset({
    "the", "a", "an", "and", "or", "but", "for", "not", "nor",
    "so", "yet", "both", "either", "neither", "page", "date",
    "invoice", "total", "subtotal", "tax", "gst",
})


# ─────────────────────────────────────────────────────────────────────────── #
#  Public API                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def validate_email(raw: str) -> tuple[str, float]:
    """Validate an email address.

    Returns:
        (cleaned_email, confidence) or ("", 0.0) if invalid.
    """
    s = raw.strip().lower()
    if _EMAIL_VAL_RE.match(s):
        return s, 0.93
    return "", 0.0


def validate_phone(raw: str) -> tuple[str, float]:
    """Validate a phone number.

    Accepts:
    - Numbers with an explicit separator (space/dash/dot) between groups.
    - Numbers with a country code (+XX) followed by a 10-digit number,
      even if there is no separator within the national number.

    Returns:
        (raw_phone, confidence) or ("", 0.0) if invalid.
    """
    s = raw.strip()
    digits = re.sub(r"\D", "", s)
    if len(digits) < 7:
        return "", 0.0
    # Accept if there is an explicit separator in the national number part.
    if _PHONE_VAL_RE.search(s):
        return s, 0.90
    # Also accept country-code numbers: +XX followed by 9–12 digits.
    if re.match(r"^\+\d{1,3}[\s\-.]?\d{9,12}$", s.replace(" ", "")):
        return s, 0.88
    # Accept if the whole string is just digits (10 or 11 digits — local number).
    if re.match(r"^\+?\d{10,11}$", digits):
        return s, 0.85
    return "", 0.0



def validate_money(raw: str) -> tuple[str, float]:
    """Validate and clean a monetary amount string.

    Returns:
        (cleaned_amount, confidence) or ("", 0.0) if no digits found.
    """
    s = raw.strip()
    m = _MONEY_VAL_RE.search(s)
    if m:
        cleaned = re.sub(r"([₹$€£¥])\s+", r"\1", m.group().strip())
        return cleaned, 0.80
    return "", 0.0


def validate_date_regex(raw: str) -> tuple[str, float]:
    """Validate a date string using the date regex.

    Returns:
        (date_string, confidence) or ("", 0.0).
    """
    s = raw.strip()
    m = _DATE_VAL_RE.search(s)
    if m:
        return m.group().strip(), 0.75
    return "", 0.0


def validate_date_spacy(raw: str, nlp) -> tuple[str, float]:
    """Validate a date using spaCy DATE entities.

    Falls back to regex validation if spaCy is unavailable.
    Returns:
        (date_string, confidence) or ("", 0.0).
    """
    if not raw or not raw.strip():
        return "", 0.0
    if nlp is not None:
        try:
            doc = nlp(raw[:500])
            for ent in doc.ents:
                if ent.label_ == "DATE":
                    return ent.text.strip(), 0.88
        except Exception:
            pass
    # Fallback to regex
    return validate_date_regex(raw)


def validate_name(raw: str, spacy_persons: list[str]) -> tuple[str, float]:
    """Validate a candidate person name.

    Checks:
    - Length 2–60 characters
    - ≥ 80% alphabetic/space characters
    - Not a job-title keyword
    - Bonus confidence if it appears in spaCy PERSON entities

    Returns:
        (cleaned_name, confidence) or ("", 0.0).
    """
    s = raw.strip()
    if not s or len(s) < 2 or len(s) > 60:
        return "", 0.0
    if re.search(r"\d", s):
        return "", 0.0
    if s.lower().startswith("http"):
        return "", 0.0
    if any(w in s.lower().split() for w in _TITLE_WORDS):
        return "", 0.0

    alpha_ratio = sum(c.isalpha() or c in " .-'" for c in s) / len(s)
    if alpha_ratio < 0.80:
        return "", 0.0

    # Higher confidence if spaCy confirms this as a PERSON entity.
    s_lower = s.lower()
    for sp in spacy_persons:
        if sp.lower() == s_lower or s_lower in sp.lower() or sp.lower() in s_lower:
            return s, 0.92
    return s, 0.65


def validate_org(raw: str, spacy_orgs: list[str]) -> tuple[str, float]:
    """Validate an organisation name.

    Returns:
        (cleaned_org, confidence) or ("", 0.0).
    """
    s = raw.strip().rstrip(".,;:")
    if not s or len(s) < 2 or len(s) > 80:
        return "", 0.0
    if s.lower() in _ORG_NOISE:
        return "", 0.0

    s_lower = s.lower()
    for org in spacy_orgs:
        if org.lower() == s_lower or s_lower in org.lower() or org.lower() in s_lower:
            return s, 0.88
    return s, 0.65


def make_field(value: str, confidence: float) -> dict:
    """Wrap a scalar field value in the standard confidence envelope.

    An empty *value* is stored as ``None`` so that consumers receive a JSON
    ``null`` rather than an empty string for fields that could not be
    extracted.  Confidence is capped at 1.0.

    Returns:
        {"value": value_or_None, "confidence": confidence}
    """
    return {
        "value": value if value else None,
        "confidence": round(min(confidence, 1.0), 3),
    }


def make_list_field(items: list[str], confidence: float) -> list[dict]:
    """Wrap a list of values in the standard confidence envelope.

    Returns:
        [{"value": item, "confidence": confidence}, ...]
    """
    c = round(min(confidence, 1.0), 3)
    return [{"value": item, "confidence": c} for item in items]


def pdf_bonus(is_pdf_layout: bool) -> float:
    """Return the confidence bonus for PDF-sourced layout data."""
    return 0.05 if is_pdf_layout else 0.0
