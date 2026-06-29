import logging
from typing import Final

from app.classifiers.base import BaseClassifier, ClassificationResult


# ---------------------------------------------------------------------------
# Weighted keyword rules.
# Structure: {category: {keyword: weight}}
# ---------------------------------------------------------------------------
WEIGHTED_KEYWORD_RULES: Final[dict[str, dict[str, int]]] = {
    "resume": {
        "education": 10,
        "skills": 10,
        "technical skills": 12,
        "experience": 10,
        "projects": 8,
        "internship": 8,
        "certifications": 6,
        "objective": 4,
        "summary": 4,
        "college": 6,
        "university": 6,
        "cgpa": 8,
        "linkedin": 6,
        "github": 6,
        "resume": 15,
        "curriculum vitae": 15,
    },
    "invoice": {
        "invoice": 15,
        "invoice number": 15,
        "gst": 10,
        "gstin": 10,
        "subtotal": 8,
        "tax": 8,
        "grand total": 12,
        "amount due": 10,
        "vendor": 8,
        "customer": 8,
        "unit price": 8,
        "quantity": 8,
        "invoice date": 10,
    },
    "contract": {
        "agreement": 15,
        "contract": 15,
        "party": 10,
        "parties": 10,
        "effective date": 12,
        "termination": 10,
        "confidentiality": 10,
        "liability": 10,
        "jurisdiction": 8,
        "governing law": 12,
        "signature": 8,
        "witness": 6,
    },
    "research_paper": {
        "abstract": 15,
        "introduction": 10,
        "methodology": 12,
        "methods": 10,
        "dataset": 8,
        "results": 10,
        "discussion": 8,
        "conclusion": 10,
        "references": 12,
        "bibliography": 10,
        "doi": 10,
        "authors": 8,
    },
    "medical_report": {
        "patient": 12,
        "doctor": 10,
        "hospital": 10,
        "diagnosis": 15,
        "symptoms": 10,
        "prescription": 10,
        "medication": 10,
        "clinical": 8,
        "treatment": 10,
        "laboratory": 8,
        "test result": 10,
        "medical report": 15,
    },
}

# ---------------------------------------------------------------------------
# Per-category minimum weighted score required to claim the category.
# Documents that do not reach the threshold for any category fall back to
# 'general_document'.
# ---------------------------------------------------------------------------
SCORE_THRESHOLDS: Final[dict[str, int]] = {
    "resume": 25,
    "invoice": 25,
    "contract": 25,
    "research_paper": 25,
    "medical_report": 25,
}

# ---------------------------------------------------------------------------
# Pre-computed maximum possible score per category (sum of all weights).
# Used to normalise the confidence value to [0.0, 1.0].
# ---------------------------------------------------------------------------
MAX_SCORES: Final[dict[str, int]] = {
    category: sum(keywords.values())
    for category, keywords in WEIGHTED_KEYWORD_RULES.items()
}

FALLBACK_CATEGORY: Final[str] = "general_document"


class RuleBasedClassifier(BaseClassifier):
    """Classify documents using weighted keyword scoring.

    For every supported category each keyword carries a configurable integer
    weight.  The text is lowercased once; keywords are matched by substring
    search.  The weighted score for a category is the sum of the weights of
    all keywords that appear in the text.  A category is only eligible to win
    if its score meets or exceeds the configured SCORE_THRESHOLDS value.
    Among eligible categories the one with the highest score wins.  If no
    category is eligible, 'general_document' is returned.

    Confidence is score / MAX_SCORES[winning_category], rounded to four
    decimal places, giving a value in [0.0, 1.0].

    No external libraries are used.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("app")

    def classify(self, text: str) -> ClassificationResult:
        """Return a weighted ClassificationResult for the given text.

        Args:
            text: Full plain text of the document.

        Returns:
            ClassificationResult with document_type, confidence (0.0–1.0),
            and the list of matched keywords for the winning category.
        """
        if not text or not text.strip():
            self.logger.debug(
                "Empty text supplied to classifier; returning fallback."
            )
            return ClassificationResult(
                document_type=FALLBACK_CATEGORY,
                confidence=0.0,
                matched_keywords=[],
            )

        normalised = text.lower()

        best_category: str = FALLBACK_CATEGORY
        best_score: int = 0
        best_matched: list[str] = []

        for category, keywords in WEIGHTED_KEYWORD_RULES.items():
            matched: dict[str, int] = {
                kw: weight for kw, weight in keywords.items() if kw in normalised
            }
            score: int = sum(matched.values())
            threshold: int = SCORE_THRESHOLDS[category]

            self.logger.debug(
                "Category '%s' — weighted score: %d, threshold: %d, "
                "matched keywords: %s",
                category,
                score,
                threshold,
                list(matched.keys()),
            )

            # Only eligible if it meets the threshold AND beats the current best.
            if score >= threshold and score > best_score:
                best_score = score
                best_category = category
                best_matched = list(matched.keys())

        # Compute normalised confidence for the winning category.
        if best_category == FALLBACK_CATEGORY:
            confidence: float = 0.0
            best_matched = []
        else:
            max_score: int = MAX_SCORES[best_category]
            confidence = round(best_score / max_score, 4) if max_score > 0 else 0.0

        self.logger.debug(
            "Classification result: '%s' (weighted score=%d, confidence=%.4f).",
            best_category,
            best_score,
            confidence,
        )

        return ClassificationResult(
            document_type=best_category,
            confidence=confidence,
            matched_keywords=best_matched,
        )
