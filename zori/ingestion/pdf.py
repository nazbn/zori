import io
from dataclasses import dataclass

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Note: page_start tracking is intentionally omitted in v1.
# When citation support is added, extract per-page character offsets
# during extraction rather than estimating after chunking.


@dataclass
class TextChunk:
    text: str
    item_key: str
    chunk_index: int


class PDFParser:
    def extract_text(self, pdf_bytes: bytes) -> str:
        """Extract full text from a PDF, page by page."""
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
        except Exception as e:
            raise ValueError(f"Failed to open PDF: {e}") from e

        text = "\n".join(pages).strip()

        if not text:
            raise ValueError("PDF has no extractable text (possibly a scanned document)")

        return text

    def chunk(
        self,
        text: str,
        item_key: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[TextChunk]:
        """Split text into overlapping chunks, each tagged with its source item key."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        raw_chunks = splitter.split_text(text)

        return [
            TextChunk(
                text=chunk,
                item_key=item_key,
                chunk_index=i,
            )
            for i, chunk in enumerate(raw_chunks)
        ]
