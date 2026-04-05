import sqlite3
from pathlib import Path

from zori.ingestion.pdf import TextChunk
from zori.ingestion.zotero import ZoteroItem

FTS_PATH = Path(".zori/fts.db")


class LexicalIndex:
    """SQLite FTS5 full-text index for BM25 / keyword / exact-phrase search.

    Two tables:
    - papers_fts  — paper-level fields: title, abstract, tags
    - chunks_fts  — chunk-level field:  chunk_text

    Populated directly from ZoteroItem and TextChunk at ingestion time.
    Intentionally independent of MetadataStore and ChromaVectorStore.
    """

    def __init__(self, path: Path = FTS_PATH):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
                item_key UNINDEXED,
                title,
                abstract,
                tags,
                tokenize = "porter unicode61"
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                item_key UNINDEXED,
                chunk_index UNINDEXED,
                chunk_text,
                tokenize = "porter unicode61"
            );
        """)
        self._conn.commit()

    def add_paper(self, item: ZoteroItem) -> None:
        """Index paper-level fields from a ZoteroItem."""
        tags = " ".join(item.tags)
        self._conn.execute(
            "INSERT INTO papers_fts(item_key, title, abstract, tags) VALUES (?, ?, ?, ?)",
            (item.key, item.title, item.abstract or "", tags),
        )
        self._conn.commit()

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        """Index chunk text from a list of TextChunks."""
        if not chunks:
            return
        self._conn.executemany(
            "INSERT INTO chunks_fts(item_key, chunk_index, chunk_text) VALUES (?, ?, ?)",
            [(c.item_key, c.chunk_index, c.text) for c in chunks],
        )
        self._conn.commit()

    def delete_item(self, item_key: str) -> None:
        """Remove all indexed data for an item (called before re-ingestion)."""
        self._conn.execute("DELETE FROM papers_fts WHERE item_key = ?", (item_key,))
        self._conn.execute("DELETE FROM chunks_fts WHERE item_key = ?", (item_key,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
