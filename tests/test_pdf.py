from unittest.mock import MagicMock, patch

import pytest

from zori.ingestion.pdf import PDFParser, TextChunk


@pytest.fixture
def parser():
    return PDFParser()


# --- extract_text ---

def test_extract_text_joins_pages(parser):
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Page one content."
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Page two content."

    mock_pdf = MagicMock()
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = [mock_page1, mock_page2]

    with patch("zori.ingestion.pdf.pdfplumber.open", return_value=mock_pdf):
        text = parser.extract_text(b"fake-pdf-bytes")

    assert "Page one content." in text
    assert "Page two content." in text


def test_extract_text_raises_on_empty(parser):
    mock_page = MagicMock()
    mock_page.extract_text.return_value = ""

    mock_pdf = MagicMock()
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = [mock_page]

    with patch("zori.ingestion.pdf.pdfplumber.open", return_value=mock_pdf):
        with pytest.raises(ValueError, match="no extractable text"):
            parser.extract_text(b"fake-pdf-bytes")


def test_extract_text_raises_on_bad_pdf(parser):
    with patch("zori.ingestion.pdf.pdfplumber.open", side_effect=Exception("corrupt")):
        with pytest.raises(ValueError, match="Failed to open PDF"):
            parser.extract_text(b"bad-bytes")


# --- chunk ---

def test_chunk_returns_text_chunks(parser):
    text = "word " * 500  # ~2500 chars
    chunks = parser.chunk(text, item_key="KEY1", chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(isinstance(c, TextChunk) for c in chunks)
    assert all(c.item_key == "KEY1" for c in chunks)


def test_chunk_indices_are_sequential(parser):
    text = "word " * 500
    chunks = parser.chunk(text, item_key="KEY1", chunk_size=100, chunk_overlap=20)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_short_text_returns_single_chunk(parser):
    text = "Short paper."
    chunks = parser.chunk(text, item_key="KEY1", chunk_size=1000, chunk_overlap=200)
    assert len(chunks) == 1
    assert chunks[0].text == "Short paper."
