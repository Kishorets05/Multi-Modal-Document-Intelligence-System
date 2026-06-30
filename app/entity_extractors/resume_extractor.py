"""Resume entity extractor — Hybrid Intelligent pipeline.

Pipeline
--------
1. LayoutEngine  (PDF real layout OR text heuristics)
2. SectionParser (section-aware block splitting)
3. spaCy NLP     (PERSON for name validation)
4. EntityRuler   (TECH_SKILL patterns for skills)
5. Rule Engine   (section-scoped field collection)
6. ValidationEngine (email, phone, name validation + confidence)

All fields are returned as ``{"value": ..., "confidence": float}``
objects (scalars) or ``[{"value": ..., "confidence": float}, ...]``
(lists).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from app.entity_extractors.base import BaseEntityExtractor
from app.entity_extractors.layout_engine import LayoutBlock
from app.entity_extractors.section_parser import SectionParser
from app.entity_extractors.validation_engine import (
    make_field,
    make_list_field,
    pdf_bonus,
    validate_email,
    validate_name,
    validate_phone,
)

# ─────────────────────────────────────────────────────────────────────────── #
#  Regex — contact info only                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

_EMAIL_RE: Final = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)
_PHONE_RE: Final = re.compile(
    r"(?<!\d)"
    r"(?:\+\d{1,3}[\s\-.]*)?"
    r"(?:\(?\d{2,5}\)?[\s\-.]*)?"
    r"\d{3,5}[\s\-.]?\d{4,5}"
    r"(?!\d)"
)

# ─────────────────────────────────────────────────────────────────────────── #
#  Known resume section headings                                              #
# ─────────────────────────────────────────────────────────────────────────── #

_RESUME_SECTIONS: Final[set[str]] = {
    "objective", "career objective",
    "summary", "professional summary",
    "profile", "professional profile", "about me",
    "technical skills", "skills", "core competencies", "key skills",
    "non technical skills", "soft skills", "other skills",
    "languages", "language",
    "experience", "work experience", "professional experience",
    "employment history", "work history", "career history",
    "internship", "internships",
    "projects", "project", "academic projects", "personal projects",
    "certifications", "certification", "certificates", "certificate",
    "achievements", "awards", "honors",
    "education", "academic background", "educational qualification",
    "references", "declaration",
    "hobbies", "interests", "activities", "extra curricular",
}

_COLLECT_MAP: Final[dict[str, str]] = {
    "technical skills": "skills", "skills": "skills",
    "core competencies": "skills", "key skills": "skills",
    "experience": "experience", "work experience": "experience",
    "professional experience": "experience",
    "employment history": "experience", "work history": "experience",
    "career history": "experience", "internship": "experience",
    "internships": "experience",
    "projects": "projects", "project": "projects",
    "academic projects": "projects", "personal projects": "projects",
    "certifications": "certifications", "certification": "certifications",
    "certificates": "certifications", "certificate": "certifications",
    "education": "education", "academic background": "education",
    "educational qualification": "education",
}

_SKILL_SUBLABEL_RE: Final = re.compile(
    r"^([A-Za-z][A-Za-z\s]{1,25}):\s*(.*)", re.IGNORECASE
)
_SOFT_SKILLS: Final[frozenset[str]] = frozenset({
    "analytical thinking", "problem solving", "problem-solving",
    "team collaboration", "quick learner", "communication",
    "time management", "leadership", "teamwork",
    "critical thinking", "attention to detail",
})
_BULLET_RE: Final = re.compile(r"^[\s•\-–·▪▸►*]+")
_TITLE_WORDS: Final[frozenset[str]] = frozenset({
    "developer", "engineer", "designer", "analyst", "manager",
    "architect", "consultant", "intern", "specialist", "officer",
})


# ─────────────────────────────────────────────────────────────────────────── #
#  Helpers                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def _strip_bullet(s: str) -> str:
    return _BULLET_RE.sub("", s).strip()


def _is_name_candidate(line: str) -> bool:
    s = line.strip()
    if not s or len(s) < 2 or len(s) > 60:
        return False
    if re.search(r"\d", s):
        return False
    if s.lower().startswith("http"):
        return False
    if _EMAIL_RE.search(s) or _PHONE_RE.search(s):
        return False
    if any(w in s.lower() for w in _TITLE_WORDS):
        return False
    alpha_ratio = sum(c.isalpha() or c in " .-'" for c in s) / len(s)
    return alpha_ratio > 0.80


def _extract_name_candidate(parser: SectionParser) -> str:
    """Find the best name candidate using font-size-sorted preamble blocks."""
    preamble = sorted(
        parser.get_preamble_blocks(),
        key=lambda b: b.font_size,
        reverse=True,
    )
    for block in preamble:
        if _is_name_candidate(block.stripped):
            return block.stripped

    # Fallback: all-caps isolated block anywhere in document.
    for block in parser.all_blocks:
        s = block.stripped
        if (
            s.isupper()
            and 5 <= len(s) <= 60
            and " " in s
            and _is_name_candidate(s)
        ):
            return s
    return ""


def _parse_skills_lines(lines: list[str]) -> list[str]:
    """Strip sub-labels, split on comma/pipe, exclude soft skills."""
    result: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        t = token.strip(" \t-–•·,")
        if not t or t.lower() in _SOFT_SKILLS or t in seen:
            return
        seen.add(t)
        result.append(t)

    for raw in lines:
        cleaned = _strip_bullet(raw)
        if not cleaned:
            continue
        m = _SKILL_SUBLABEL_RE.match(cleaned)
        if m:
            values = m.group(2).strip()
            if values:
                for part in re.split(r"[,|]", values):
                    _add(part)
        else:
            for part in re.split(r"[,|]", cleaned):
                _add(part)
    return result


def _parse_list_lines(lines: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        c = _strip_bullet(raw)
        if c and c not in seen:
            seen.add(c)
            result.append(c)
    return result


# ─────────────────────────────────────────────────────────────────────────── #
#  Main extractor                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class ResumeEntityExtractor(BaseEntityExtractor):
    """Extract structured entities from a resume.

    Pipeline
    --------
    1. Layout (PDF or text) + SectionParser.
    2. spaCy PERSON entities — used to validate the candidate name.
    3. EntityRuler TECH_SKILL — primary source for skills list.
    4. Sub-label parsing — augments skills from section text.
    5. Regex — email and phone only.
    6. ValidationEngine — per-field confidence wrapping.
    """

    _HEADING_NAMES: set[str] = _RESUME_SECTIONS

    def extract(self, text: str) -> dict:
        is_pdf = self._is_pdf_layout()
        layout_bonus = pdf_bonus(is_pdf)

        # ── spaCy pass ────────────────────────────────────────────────────
        doc = self._doc(text)
        spacy_persons = self._ents(doc, "PERSON")

        # EntityRuler TECH_SKILL entities.
        tech_skills_from_ruler: list[str] = []
        if doc is not None:
            seen_skills: set[str] = set()
            for ent in doc.ents:
                if ent.label_ == "TECH_SKILL":
                    norm = ent.text.strip()
                    if norm and norm not in seen_skills:
                        seen_skills.add(norm)
                        tech_skills_from_ruler.append(norm)

        # ── Layout + section parser ───────────────────────────────────────
        parser = self._build_parser(text)

        # ── Contact info (regex) ──────────────────────────────────────────
        email_raw = ""
        em = _EMAIL_RE.search(text)
        if em:
            email_raw = em.group()

        phone_raw = ""
        for pm in _PHONE_RE.finditer(text):
            digits = re.sub(r"\D", "", pm.group())
            if len(digits) >= 7:
                phone_raw = pm.group().strip()
                break

        # ── Name ──────────────────────────────────────────────────────────
        name_candidate = _extract_name_candidate(parser)
        name_value, name_conf = validate_name(name_candidate, spacy_persons)
        name_conf = min(name_conf + layout_bonus, 1.0)

        # ── Skills ────────────────────────────────────────────────────────
        # Primary: EntityRuler hits (confidence 0.85)
        # Augment: sub-label parse from skills section (confidence 0.65 for
        # items not recognised by the ruler).
        ruler_set = {s.lower() for s in tech_skills_from_ruler}

        skill_blocks = parser.extract_between(
            "technical skills", stop_before=["certifications"]
        )
        section_skills = _parse_skills_lines([b.text for b in skill_blocks])

        merged_skills: list[dict] = []
        merged_seen: set[str] = set()

        # EntityRuler skills first (higher confidence).
        for s in tech_skills_from_ruler:
            key = s.lower()
            if key not in merged_seen:
                merged_seen.add(key)
                merged_skills.append({"value": s, "confidence": round(0.85 + layout_bonus, 3)})

        # Section skills not caught by the ruler.
        for s in section_skills:
            key = s.lower()
            if key not in merged_seen:
                merged_seen.add(key)
                merged_skills.append({"value": s, "confidence": round(0.65 + layout_bonus, 3)})

        # ── Section-parsed lists (experience, projects, certs, education) ─
        exp_blocks = parser.extract_between("experience", stop_before=["projects"])
        proj_blocks = parser.extract_between("projects", stop_before=["non technical skills"])
        cert_blocks = parser.extract_between("certifications", stop_before=["education"])
        edu_blocks  = parser.extract_between("education", stop_before=["languages"])

        section_conf = round(0.75 + layout_bonus, 3)

        # ── Validate contact fields ───────────────────────────────────────
        email_val, email_conf = validate_email(email_raw)
        phone_val, phone_conf = validate_phone(phone_raw)

        return {
            "name":           make_field(name_value, name_conf),
            "email":          make_field(email_val, email_conf),
            "phone":          make_field(phone_val, phone_conf),
            "skills":         merged_skills,
            "experience":     make_list_field(
                                  _parse_list_lines([b.text for b in exp_blocks]),
                                  section_conf,
                              ),
            "projects":       make_list_field(
                                  _parse_list_lines([b.text for b in proj_blocks]),
                                  section_conf,
                              ),
            "certifications": make_list_field(
                                  _parse_list_lines([b.text for b in cert_blocks]),
                                  section_conf,
                              ),
            "education":      make_list_field(
                                  _parse_list_lines([b.text for b in edu_blocks]),
                                  section_conf,
                              ),
        }
