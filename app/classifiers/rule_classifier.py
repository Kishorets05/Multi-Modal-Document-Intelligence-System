import logging
from typing import Final

from app.classifiers.base import BaseClassifier


# Each entry is (category_name, tuple_of_keywords).
# Order defines tiebreak priority: earlier entries win ties.
# Keywords are matched case-insensitively against the full document text.
KEYWORD_RULES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (
        "resume",
        (
            "education",
            "skills",
            "technical skills",
            "experience",
            "work experience",
            "internship",
            "projects",
            "certifications",
            "objective",
            "summary",
            "profile",
            "languages",
            "achievements",
            "cgpa",
            "college",
            "university",
            "bachelor",
            "master",
            "resume",
            "curriculum vitae",
            "cv",
        ),
    ),
    (
        "invoice",
        (
            "invoice",
            "invoice number",
            "invoice no",
            "bill to",
            "ship to",
            "gst",
            "gstin",
            "tax",
            "subtotal",
            "total",
            "total amount",
            "amount due",
            "quantity",
            "unit price",
            "payment terms",
            "invoice date",
            "due date",
        ),
    ),
    (
        "research_paper",
        (
            "abstract",
            "introduction",
            "literature review",
            "methodology",
            "methods",
            "experiment",
            "experimental setup",
            "dataset",
            "results",
            "discussion",
            "conclusion",
            "references",
            "bibliography",
            "authors",
        ),
    ),
    (
        "contract",
        (
            "agreement",
            "contract",
            "party",
            "parties",
            "terms",
            "conditions",
            "liability",
            "confidentiality",
            "termination",
            "effective date",
            "governing law",
            "signature",
            "witness",
        ),
    ),
    (
        "medical_report",
        (
            "patient",
            "doctor",
            "hospital",
            "diagnosis",
            "symptoms",
            "prescription",
            "medication",
            "clinical",
            "treatment",
            "laboratory",
            "blood pressure",
            "medical report",
        ),
    ),
)

FALLBACK_CATEGORY: Final[str] = "general_document"


class RuleBasedClassifier(BaseClassifier):
    """Classify documents using deterministic keyword frequency scoring.

    For each supported category a set of keywords is defined. The text is
    lowercased once and each keyword is checked for membership. The category
    with the highest keyword hit-count wins. Ties are resolved by the
    definition order in KEYWORD_RULES. If no category scores at least one
    hit the fallback 'general_document' is returned.

    No external libraries are used. Classification is O(n * k) where n is
    the text length and k is the total number of keywords.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("app")

    def classify(self, text: str) -> str:
        """Return the best-matching document category for the given text.

        Args:
            text: Full plain text of the document.

        Returns:
            A category string from BaseClassifier.SUPPORTED_CATEGORIES.
        """
        if not text or not text.strip():
            self.logger.debug("Empty text supplied to classifier; returning fallback.")
            return FALLBACK_CATEGORY

        normalised = text.lower()

        best_category = FALLBACK_CATEGORY
        best_score = 0

        for category, keywords in KEYWORD_RULES:
            score = sum(1 for kw in keywords if kw in normalised)
            self.logger.debug(
                "Category '%s' scored %d keyword hits.", category, score
            )
            # Strict greater-than preserves tiebreak priority of earlier entries.
            if score > best_score:
                best_score = score
                best_category = category

        self.logger.debug(
            "Classification result: '%s' (score=%d).", best_category, best_score
        )
        return best_category
