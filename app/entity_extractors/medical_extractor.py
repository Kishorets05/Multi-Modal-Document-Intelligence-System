"""Medical report entity extractor — Hybrid Intelligent pipeline.

Pipeline
--------
1. Section map  (heading-based, same catalogue as before)
2. spaCy NLP    (PERSON for patient and doctor, ORG for hospital)
3. Rule Engine  (label:value patterns, section-scoped then full-doc)
4. Regex        (dose units, lab values — unchanged)
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
    validate_org,
)

# ─────────────────────────────────────────────────────────────────────────── #
#  Medical section heading catalogue                                          #
# ─────────────────────────────────────────────────────────────────────────── #

_MED_SECTIONS: Final[dict[str, str]] = {
    "patient information":               "patient",
    "patient details":                   "patient",
    "patient":                           "patient",
    "personal information":              "patient",
    "physician information":             "doctor",
    "doctor":                            "doctor",
    "attending physician":               "doctor",
    "referring physician":               "doctor",
    "consultant":                        "doctor",
    "hospital":                          "hospital",
    "facility":                          "hospital",
    "clinic":                            "hospital",
    "institution":                       "hospital",
    "chief complaint":                   "complaint",
    "presenting complaint":              "complaint",
    "diagnosis":                         "diagnosis",
    "diagnoses":                         "diagnosis",
    "assessment":                        "diagnosis",
    "clinical impression":               "diagnosis",
    "impression":                        "diagnosis",
    "final diagnosis":                   "diagnosis",
    "primary diagnosis":                 "diagnosis",
    "differential diagnosis":            "diagnosis",
    "medications":                       "medications",
    "medication":                        "medications",
    "current medications":               "medications",
    "prescriptions":                     "medications",
    "prescription":                      "medications",
    "drugs":                             "medications",
    "treatment plan":                    "medications",
    "laboratory results":                "tests",
    "lab results":                       "tests",
    "test results":                      "tests",
    "investigations":                    "tests",
    "findings":                          "tests",
    "investigation":                     "tests",
    "pathology":                         "tests",
    "physical examination":              "exam",
    "examination":                       "exam",
    "vitals":                            "exam",
    "vital signs":                       "exam",
    "history of present illness":        "history",
    "history":                           "history",
    "past medical history":              "history",
    "past history":                      "history",
    "social history":                    "history",
    "family history":                    "history",
    "allergies":                         "allergies",
    "treatment":                         "treatment",
    "management":                        "treatment",
    "plan":                              "treatment",
    "follow up":                         "followup",
    "follow-up":                         "followup",
    "recommendations":                   "followup",
    "discharge summary":                 "summary",
    "summary":                           "summary",
    "billing":                           "__ignore__",
    "billing information":               "__ignore__",
    "disclaimer":                        "__ignore__",
    "footer":                            "__ignore__",
    "advertisement":                     "__ignore__",
}

_SECTION_HEAD_RE: Final = re.compile(
    r"^\s*("
    + "|".join(re.escape(h) for h in sorted(_MED_SECTIONS, key=len, reverse=True))
    + r")\s*[:\-]?\s*$",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────── #
#  Regex helpers — dose units and lab values                                  #
# ─────────────────────────────────────────────────────────────────────────── #

_DOSE_UNIT_RE: Final = re.compile(
    r"\b\d+\s*(?:mg|mcg|μg|ml|mL|g|kg|units?|IU|tabs?|capsules?|cap|"
    r"drops?|tsp|tbsp|puff|spray|patch|sachet|vial|tablet)\b",
    re.IGNORECASE,
)
_LAB_VALUE_RE: Final = re.compile(
    r"^(.{3,50}?)\s*[:\-]\s*"
    r"(\d+\.?\d*)\s*"
    r"(mg/dL|mmHg|g/dL|%|U/L|IU/L|mEq/L|mmol/L|µmol/L|ng/mL|pg/mL|"
    r"cells/µL|10\^3/µL|10\^6/µL|mIU/L|nmol/L|pmol/L|bpm|rpm)?"
    r"(?:\s*\(.*?\))?\s*$",
    re.IGNORECASE,
)

# Label patterns (scalar fields — section-scoped then full doc)
_PATIENT_LABEL_RE: Final = re.compile(
    r"(?:patient(?:'s)?\s+name|name\s+of\s+patient|name)\s*[:\-]\s*([^\n,;|]{2,60})",
    re.IGNORECASE,
)
_DOCTOR_LABEL_RE: Final = re.compile(
    r"(?:dr\.?\s+|doctor\s*[:\-]\s*|physician\s*[:\-]\s*|"
    r"consultant\s*[:\-]\s*|attending\s+physician\s*[:\-]\s*)"
    r"([A-Z][a-zA-Z\s\.]{2,50})",
    re.IGNORECASE,
)
_HOSPITAL_LABEL_RE: Final = re.compile(
    r"(?:hospital|clinic|health\s+cent(?:er|re)|medical\s+cent(?:er|re)|"
    r"institute|facility)\s*[:\-]\s*([^\n,;]{3,80})",
    re.IGNORECASE,
)
_DIAGNOSIS_LABEL_RE: Final = re.compile(
    r"(?:diagnosis|diagnoses|clinical\s+impression|impression|"
    r"assessment|final\s+diagnosis|primary\s+diagnosis|diagnosed\s+with)"
    r"\s*[:\-]\s*([^\n.;]{3,120})",
    re.IGNORECASE,
)

# "Dr. Smith" prefix pattern for spaCy PERSON validation
_DR_PREFIX_RE: Final = re.compile(
    r"\bdr\.?\s+([A-Z][a-zA-Z\s\.]{2,50})", re.IGNORECASE
)


def _is_section_head(line: str) -> bool:
    return bool(_SECTION_HEAD_RE.match(line))


def _norm_head(line: str) -> str:
    return line.strip().rstrip(":- ").strip().lower()


def _clean(s: str) -> str:
    return s.strip().rstrip(".,;:")


def _extract_scalar(
    label_re: re.Pattern, section_text: str, full_text: str
) -> str:
    m = label_re.search(section_text) if section_text else None
    if not m:
        m = label_re.search(full_text)
    return _clean(m.group(1)) if m else ""


# ─────────────────────────────────────────────────────────────────────────── #
#  Main extractor                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class MedicalEntityExtractor(BaseEntityExtractor):
    """Extract structured entities from a medical report.

    Priority chain per field
    ------------------------
    patient_name : spaCy PERSON on patient section → regex label:value
    doctor       : spaCy PERSON near "Dr." → regex label:value
    hospital     : spaCy ORG on hospital section → regex label:value
    diagnosis    : regex label:value (no good NER label)
    medications  : section lines + dose-unit regex scan (unchanged)
    test_results : lab-value regex on test section lines (unchanged)
    """

    def extract(self, text: str) -> dict:
        is_pdf = self._is_pdf_layout()
        layout_bonus = pdf_bonus(is_pdf)

        # ── spaCy pass ────────────────────────────────────────────────────
        doc = self._doc(text)
        spacy_persons = self._ents(doc, "PERSON")
        spacy_orgs    = self._ents(doc, "ORG")

        # ── Build section map ─────────────────────────────────────────────
        lines = [ln.rstrip() for ln in text.splitlines()]
        sections: dict[str, list[str]] = {"__preamble__": []}
        current_key = "__preamble__"

        for line in lines:
            stripped = line.strip()
            if _is_section_head(stripped):
                mapped = _MED_SECTIONS.get(_norm_head(stripped), "__unknown__")
                current_key = mapped
                sections.setdefault(current_key, [])
            else:
                sections.setdefault(current_key, []).append(stripped)

        def _sec_text(*keys: str) -> str:
            for k in keys:
                if k in sections:
                    return "\n".join(sections[k])
            return ""

        patient_text = _sec_text("patient")
        doctor_text  = _sec_text("doctor")
        hosp_text    = _sec_text("hospital")
        diag_text    = _sec_text("diagnosis")
        med_lines    = sections.get("medications", [])
        test_lines   = sections.get("tests", [])

        # ── Pre-compute doctor-name candidates ────────────────────────────────────
        # Collecting all "Dr. X" tokens before patient extraction prevents a
        # doctor being accidentally assigned to the patient field when the
        # doctor's name appears first among spaCy PERSON entities.
        _dr_names_lower: set[str] = {
            m.group(1).strip().lower()
            for m in _DR_PREFIX_RE.finditer(text)
        }

        # ── Patient name ──────────────────────────────────────────────────────────
        patient_val  = ""
        patient_conf = 0.0

        if spacy_persons:
            # 1st preference: PERSON in the patient section that is not a doctor.
            for p in spacy_persons:
                if any(d in p.lower() for d in _dr_names_lower):
                    continue
                if patient_text and p.lower() in patient_text.lower():
                    patient_val  = p
                    patient_conf = 0.92 + layout_bonus
                    break
            # 2nd preference: any non-doctor PERSON from the full document.
            if not patient_val:
                for p in spacy_persons:
                    if not any(d in p.lower() for d in _dr_names_lower):
                        patient_val  = p
                        patient_conf = 0.85 + layout_bonus
                        break

        if not patient_val:
            patient_val  = _extract_scalar(_PATIENT_LABEL_RE, patient_text, text)
            patient_conf = 0.72 + layout_bonus if patient_val else 0.0

        # ── Doctor ────────────────────────────────────────────────────────
        doctor_val  = ""
        doctor_conf = 0.0

        # Prefer PERSON entity that appears after "Dr." prefix
        dr_match = _DR_PREFIX_RE.search(text)
        if dr_match:
            cand = dr_match.group(1).strip()
            for p in spacy_persons:
                if p.lower() in cand.lower() or cand.lower() in p.lower():
                    doctor_val  = cand
                    doctor_conf = 0.90 + layout_bonus
                    break
            if not doctor_val:
                doctor_val  = cand
                doctor_conf = 0.78 + layout_bonus

        if not doctor_val:
            doctor_val  = _extract_scalar(_DOCTOR_LABEL_RE, doctor_text, text)
            doctor_conf = 0.72 + layout_bonus if doctor_val else 0.0

        if doctor_val:
            doctor_val = doctor_val.rstrip(".,")

        # ── Hospital ──────────────────────────────────────────────────────
        hospital_val  = ""
        hospital_conf = 0.0

        # spaCy ORG on hospital section text
        if hosp_text.strip() and spacy_orgs:
            hospital_val  = spacy_orgs[0]
            hospital_conf = 0.88 + layout_bonus
        else:
            hospital_val  = _extract_scalar(_HOSPITAL_LABEL_RE, hosp_text, text)
            hospital_conf = 0.70 + layout_bonus if hospital_val else 0.0

        # Validate with org validator
        if hospital_val:
            v, c = validate_org(hospital_val, spacy_orgs)
            if v:
                hospital_val  = v
                hospital_conf = max(hospital_conf, c + layout_bonus)

        # ── Diagnosis ─────────────────────────────────────────────────────
        diagnosis_val  = _extract_scalar(_DIAGNOSIS_LABEL_RE, diag_text, text)
        diagnosis_conf = round(0.75 + layout_bonus, 3) if diagnosis_val else 0.0

        # ── Medications ───────────────────────────────────────────────────
        seen_meds: set[str] = set()
        meds_raw: list[str] = []

        for raw in med_lines:
            c = raw.strip()
            if c and not _is_section_head(c) and c not in seen_meds:
                seen_meds.add(c)
                meds_raw.append(c)

        for raw in lines:
            c = raw.strip()
            if _DOSE_UNIT_RE.search(c) and c not in seen_meds:
                seen_meds.add(c)
                meds_raw.append(c)

        med_conf = round(0.78 + layout_bonus, 3)

        # ── Test results ──────────────────────────────────────────────────
        seen_tests: set[str] = set()
        tests_raw: list[str] = []

        for raw in test_lines:
            c = raw.strip()
            if c and not _is_section_head(c) and _LAB_VALUE_RE.match(c) and c not in seen_tests:
                seen_tests.add(c)
                tests_raw.append(c)

        test_conf = round(0.80 + layout_bonus, 3)

        return {
            "patient_name": make_field(patient_val,  round(min(patient_conf, 1.0), 3)),
            "doctor":       make_field(doctor_val,   round(min(doctor_conf, 1.0), 3)),
            "hospital":     make_field(hospital_val, round(min(hospital_conf, 1.0), 3)),
            "diagnosis":    make_field(diagnosis_val, diagnosis_conf),
            "medications":  make_list_field(meds_raw, med_conf),
            "test_results": make_list_field(tests_raw, test_conf),
        }
