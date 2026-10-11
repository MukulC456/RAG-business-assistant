"""
Supports PDF, TXT or DOCX files
How it works: The loader preserves the filename and PDF page number. Later, our answer generator will use this metadata to produce citations.
Note that scanned PDFs may need OCR, which we can add later.
"""

from io import BytesIO
from pathlib import Path

from docx import Document
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def load_document(
    file_bytes: bytes,
    filename: str,
) -> list[dict]:
    """
    Extract text from a supported document.

    Returns a list of pages or sections with source metadata.
    PDF page numbers are one-based.
    """
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            "Upload PDF, TXT, or DOCX files."
        )

    if extension == ".pdf":
        reader = PdfReader(BytesIO(file_bytes))

        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if text:
                pages.append({
                    "text": text,
                    "source": filename,
                    "page": page_number,
                })

        return pages

    if extension == ".txt":
        text = file_bytes.decode("utf-8-sig").strip()

        return [{
            "text": text,
            "source": filename,
            "page": None,
        }] if text else []

    if extension == ".docx":
        document = Document(BytesIO(file_bytes))

        paragraphs = [
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        text = "\n".join(paragraphs)

        return [{
            "text": text,
            "source": filename,
            "page": None,
        }] if text else []

    return []
