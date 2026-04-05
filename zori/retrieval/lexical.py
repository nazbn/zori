import re
import sqlite3
from pathlib import Path

from zori.ingestion.pdf import TextChunk
from zori.ingestion.zotero import ZoteroItem

DB_PATH = Path(".zori/zori.db")


class LexicalIndex:
    """SQLite FTS5 full-text index for BM25 / keyword / exact-phrase search.

    Two tables:
    - papers_fts  — paper-level fields: title, abstract, tags
    - chunks_fts  — chunk-level field:  chunk_text

    Populated directly from ZoteroItem and TextChunk at ingestion time.
    Intentionally independent of MetadataStore and ChromaVectorStore.
    """

    def __init__(self, path: Path = DB_PATH):
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

    @staticmethod
    def _fts_query(text: str) -> str:
        """Sanitize plain text into a safe FTS5 query.

        Replaces FTS5 metacharacters (hyphens, quotes, operators) with spaces
        so that compound terms like 'super-resolution' match both tokens.
        """
        return re.sub(r'[-"^*()\[\]{}|!:]', ' ', text).strip()

    def search_papers(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """BM25 on paper-level fields (title weighted 10x, abstract 5x, tags 2x).

        Returns (item_key, bm25_score) sorted best-first.
        FTS5 BM25 returns negative values; more negative = better match.
        """
        fts_q = self._fts_query(query)
        if not fts_q:
            return []
        try:
            rows = self._conn.execute(
                "SELECT item_key, bm25(papers_fts, 0.0, 10.0, 5.0, 2.0) AS score "
                "FROM papers_fts WHERE papers_fts MATCH ? ORDER BY score LIMIT ?",
                (fts_q, top_k),
            ).fetchall()
            return [(r["item_key"], r["score"]) for r in rows]
        except Exception:
            return []

    def search_chunks(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """BM25 on chunk text; returns the best-scoring chunk per paper."""
        fts_q = self._fts_query(query)
        if not fts_q:
            return []
        try:
            rows = self._conn.execute(
                "SELECT item_key, bm25(chunks_fts) AS score "
                "FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                (fts_q, top_k * 3),
            ).fetchall()
        except Exception:
            return []
        best: dict[str, float] = {}
        for r in rows:
            key, score = r["item_key"], r["score"]
            if key not in best or score < best[key]:  # more negative = better
                best[key] = score
        return sorted(best.items(), key=lambda x: x[1])[:top_k]

    def search_title(self, title: str, top_k: int = 5) -> list[str]:
        """High-precision FTS5 lookup scoped to the title column only."""
        escaped = title.replace('"', '""')
        try:
            rows = self._conn.execute(
                "SELECT item_key FROM papers_fts WHERE papers_fts MATCH ? "
                "ORDER BY bm25(papers_fts, 0.0, 100.0, 0.0, 0.0) LIMIT ?",
                (f'title : "{escaped}"', top_k),
            ).fetchall()
            return [r["item_key"] for r in rows]
        except Exception:
            return []

    def get_full_text(self, item_key: str) -> str:
        """Reconstruct the full paper text from stored chunks, ordered by chunk_index."""
        rows = self._conn.execute(
            "SELECT chunk_text FROM chunks_fts WHERE item_key = ? ORDER BY chunk_index",
            (item_key,),
        ).fetchall()
        return "\n\n".join(r["chunk_text"] for r in rows)

    def close(self) -> None:
        self._conn.close()
