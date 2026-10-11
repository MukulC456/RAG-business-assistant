
from io import BytesIO

from docx import Document

from app.ingestion.loaders import load_document
from app.ingestion.chunking import chunk_text, chunk_documents


def test_chunk_text_returns_overlapping_chunks():
    text = "A" * 100

    chunks = chunk_text(
        text,
        chunk_size=40,
        chunk_overlap=10,
    )

    assert len(chunks) == 4
    assert chunks[0][-10:] == chunks[1][:10]


def test_chunk_text_empty_input():
    assert chunk_text("") == []


def test_chunk_documents_preserves_metadata():
    documents = [{
        "text": "Refunds are available within 30 days.",
        "source": "policy.txt",
        "page": None,
    }]

    chunks = chunk_documents(documents)

    assert chunks[0]["source"] == "policy.txt"
    assert chunks[0]["page"] is None
    assert chunks[0]["chunk_index"] == 0


def test_load_txt():
    result = load_document(
        b"Refunds are available within 30 days.",
        "policy.txt",
    )

    assert result[0]["text"] == (
        "Refunds are available within 30 days."
    )


def test_load_docx():
    document = Document()
    document.add_paragraph("Company refund policy")

    buffer = BytesIO()
    document.save(buffer)

    result = load_document(
        buffer.getvalue(),
        "policy.docx",
    )

    assert result[0]["text"] == "Company refund policy"

"""
Run the tests:
    python -m pytest tests/test_ingestion.py -v
The tests should pass before we move forward. The embedding model test is deliberately separate because its first run downloads model files and takes longer.
"""