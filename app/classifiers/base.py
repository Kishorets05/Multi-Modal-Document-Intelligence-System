from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    """Abstract base for all document classifiers.

    All concrete classifiers must implement the classify method and return
    one of the supported category strings.
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
    def classify(self, text: str) -> str:
        """Classify the document text and return its category.

        Args:
            text: Full plain text of the document, as extracted by Module 3.

        Returns:
            One of the supported category strings:
            'resume', 'invoice', 'research_paper', 'contract',
            'medical_report', or 'general_document'.
        """
