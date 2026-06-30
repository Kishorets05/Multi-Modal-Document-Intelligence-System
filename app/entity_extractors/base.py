"""Abstract base for all document-type-specific entity extractors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.entity_extractors.layout_engine import LayoutEngine
from app.entity_extractors.section_parser import SectionParser


class BaseEntityExtractor(ABC):
    """Abstract base for all document-type-specific entity extractors.

    Every concrete extractor must implement a single ``extract`` method that
    accepts the full plain-text content of a document and returns a dict of
    structured entities.  The shape of that dict is defined by each subclass.

    The optional ``workspace_dir`` is injected by ``EntityExtractionService``
    and enables the layout engine to open the original PDF for real layout
    data (font sizes, coordinates, bold flags).  When ``None`` the engine
    falls back to heuristic text analysis.

    Design rules
    ------------
    - Return empty strings / empty lists for fields that cannot be found.
    - Never raise; absorb per-field errors silently and continue.
    - No external AI/ML APIs — only PyMuPDF, regex, and string processing.
    """

    # Subclasses must declare their known section headings so the
    # SectionParser does not split on ALL-CAPS body text.
    _HEADING_NAMES: set[str] = set()

    def __init__(self, workspace_dir: Path | None = None) -> None:
        self._workspace_dir: Path | None = (
            Path(workspace_dir) if workspace_dir else None
        )

    @abstractmethod
    def extract(self, text: str) -> dict:
        """Extract structured entities from the document plain text.

        Args:
            text: Full plain text read from ``extracted_text.txt``.

        Returns:
            A dict whose keys are the entity field names and whose values
            are strings, lists of strings, or integers — never None.
        """

    # ── shared helpers available to every subclass ─────────────────────── #

    def _build_parser(self, text: str) -> SectionParser:
        """Build a ``SectionParser`` using PDF layout when available."""
        if self._workspace_dir is not None:
            engine = LayoutEngine.from_workspace(self._workspace_dir, text)
        else:
            engine = LayoutEngine.from_text(text)

        return SectionParser(
            engine,
            heading_names=self._HEADING_NAMES if self._HEADING_NAMES else None,
        )

    def _nlp(self):
        """Return the shared spaCy Language object, or None if unavailable."""
        try:
            from app.entity_extractors.nlp_pipeline import get_nlp
            return get_nlp()
        except Exception:
            return None

    def _doc(self, text: str):
        """Return a spaCy Doc for *text*, or None if spaCy is unavailable.

        Caps input at 100 000 characters to avoid memory spikes on very long
        documents.
        """
        nlp = self._nlp()
        if nlp is None:
            return None
        try:
            return nlp(text[:100_000])
        except Exception:
            return None

    def _ents(self, doc, label: str) -> list[str]:
        """Return deduplicated entity texts for the given NER label.

        Args:
            doc:   A spaCy Doc object (may be None).
            label: NER label string, e.g. ``"PERSON"``, ``"ORG"``.

        Returns:
            Ordered list of unique entity text strings (empty if doc is None).
        """
        if doc is None:
            return []
        seen: set[str] = set()
        result: list[str] = []
        for ent in doc.ents:
            if ent.label_ == label and ent.text not in seen:
                seen.add(ent.text)
                result.append(ent.text)
        return result

    def _is_pdf_layout(self) -> bool:
        """Return True if the workspace contains an original PDF file."""
        if self._workspace_dir is None:
            return False
        pdf = self._workspace_dir / "original.pdf"
        return pdf.is_file()
