from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ClassificationResult:
    """Full output of a single document classification run.

    Attributes:
        document_type: The winning category string.  Always one of the six
            supported classes: 'resume', 'invoice', 'research_paper',
            'contract', 'medical_report', or 'general_document'.
        confidence: Normalised score in [0.0, 1.0].  Computed as the
            category's weighted score divided by its theoretical maximum.
            0.0 for 'general_document' (no category met its threshold).
        matched_keywords: Keywords from the winning category that were found
            in the document text.  Empty list for 'general_document'.
    """

    document_type: str
    confidence: float
    matched_keywords: list[str] = field(default_factory=list)


class BaseClassifier(ABC):
    """Abstract base for all document classifiers.

    All concrete classifiers must implement the classify method and return
    a ClassificationResult.
    """

    SUPPORTED_CATEGORIES = frozenset(
        {
            "resume",
            "invoice",
            "research_paper",
            "contract",
            "medical_report",
            "general_document",
        }
    )

    @abstractmethod
    def classify(self, text: str) -> ClassificationResult:
        """Classify the document text and return a ClassificationResult.

        Args:
            text: Full plain text of the document, as extracted by Module 3.

        Returns:
            ClassificationResult with document_type, confidence (0.0–1.0),
            and matched_keywords.
        """
