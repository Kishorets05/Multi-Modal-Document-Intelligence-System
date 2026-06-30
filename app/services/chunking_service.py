"""ChunkingService — Module 6 orchestration layer.

Follows the exact same service pattern as ``EntityExtractionService`` and
``DocumentClassifierService``:

* One ``__init__`` that accepts ``upload_dir`` (defaults to settings).
* One primary method that reads from the workspace and returns a plain dict.
* A module-level exception class for all domain errors.
* No HTTP concerns — those live in the API layer.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.chunkers.engine import ChunkingEngine
from app.chunkers.models import Chunk
from app.config.settings import settings


class ChunkingError(Exception):
    """Raised when chunking cannot proceed for a document.

    Covers:
    - Missing ``extracted_text.txt``.
    - Empty extracted text (nothing to chunk).
    - Missing or unreadable ``metadata.json``.
    """


class ChunkingService:
    """Orchestrate intelligent chunking for an uploaded, classified document.

    Workflow
    --------
    1. Read ``extracted_text.txt`` from the document workspace.
    2. Read ``metadata.json`` to obtain ``document_type``.
    3. Instantiate ``ChunkingEngine`` with the workspace path so that the
       engine can access the original PDF for layout-aware chunking.
    4. Return the ordered list of chunk dicts.

    The service is idempotent — calling it multiple times for the same
    document produces the same result (chunks are not persisted to disk).
    """

    def __init__(
        self,
        upload_dir: Path = settings.UPLOAD_DIR,
        max_tokens: int = 400,
        overlap_tokens: int = 50,
    ) -> None:
        self.upload_dir = Path(upload_dir)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.logger = logging.getLogger("app")

    def chunk_document(self, document_id: str) -> dict:
        """Chunk the document and return structured results.

        Args:
            document_id: The unique document identifier (workspace folder name).

        Returns:
            dict with keys:
                document_id   (str)        — echoed from the argument.
                document_type (str)        — as stored in metadata.json.
                chunk_count   (int)        — total number of chunks produced.
                chunks        (list[dict]) — ordered chunk objects.

        Raises:
            ChunkingError: On missing files, empty text, or unreadable
                metadata.json.
        """
        workspace_dir = self.upload_dir / document_id
        text_file = workspace_dir / "extracted_text.txt"
        metadata_file = workspace_dir / "metadata.json"

        self.logger.info(
            "Chunking started — document_id: '%s'", document_id
        )

        # ── Read extracted text ─────────────────────────────────────────────
        if not text_file.is_file():
            raise ChunkingError(
                f"extracted_text.txt not found for document '{document_id}'. "
                "Run text extraction before chunking."
            )

        try:
            text = text_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise ChunkingError(
                f"Failed to read extracted_text.txt for document '{document_id}'."
            ) from exc

        if not text.strip():
            raise ChunkingError(
                f"extracted_text.txt is empty for document '{document_id}'. "
                "Cannot chunk a document with no text."
            )

        # ── Read metadata ───────────────────────────────────────────────────
        if not metadata_file.is_file():
            raise ChunkingError(
                f"metadata.json not found for document '{document_id}'."
            )

        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ChunkingError(
                f"Failed to read metadata.json for document '{document_id}'."
            ) from exc

        document_type: str = metadata.get("document_type") or "general_document"

        # ── Chunk ───────────────────────────────────────────────────────────
        engine = ChunkingEngine(
            workspace_dir=workspace_dir,
            document_id=document_id,
            document_type=document_type,
            max_tokens=self.max_tokens,
            overlap_tokens=self.overlap_tokens,
        )
        chunks: list[Chunk] = engine.chunk(text)

        self.logger.info(
            "Chunking completed — document_id: '%s', document_type: '%s', "
            "chunks: %d",
            document_id,
            document_type,
            len(chunks),
        )

        return {
            "document_id": document_id,
            "document_type": document_type,
            "chunk_count": len(chunks),
            "chunks": [c.to_dict() for c in chunks],
        }
