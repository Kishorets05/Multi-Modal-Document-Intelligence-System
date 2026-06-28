from pathlib import Path
from typing import Any

from PIL import Image


class ImageReader:
    """Extract metadata from an image file."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = Path(file_path)

    def read(self) -> dict[str, Any]:
        """Return extracted image metadata."""
        with Image.open(self.file_path) as image:
            width, height = image.size
            img_format = image.format
            mode = image.mode
            dpi = image.info.get("dpi") if image.info else None

        return {
            "document_type": "image",
            "page_count": None,
            "paragraph_count": None,
            "image_width": width,
            "image_height": height,
            "document_size": self.file_path.stat().st_size,
            "document_metadata": {
                "format": img_format,
                "mode": mode,
                "dpi": dpi,
            },
            "reader_used": self.__class__.__name__,
        }
