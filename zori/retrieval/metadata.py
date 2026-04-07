import json
import sqlite3
from pathlib import Path

from zori.ingestion.zotero import ZoteroItem

DB_PATH = Path(".zori/zori.db")


class MetadataStore:
    """Persists paper-level metadata in a SQLite table keyed by Zotero item key."""

    def __init__(self, path: Path = DB_PATH):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                item_key TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                year TEXT,
                journal TEXT,
                doi TEXT,
                item_type TEXT,
                authors TEXT NOT NULL DEFAULT '[]',
                tags TEXT NOT NULL DEFAULT '[]',
                attachment_keys TEXT NOT NULL DEFAULT '[]'
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year)")
        self._conn.commit()

    def save(self, item: ZoteroItem) -> None:
        # Store all PDF attachment keys. v1 uses the first for download.
        # TODO: smarter attachment selection when multiple PDFs exist (future release)
        attachment_keys = [a.key for a in item.attachments]
        self._conn.execute("""
            INSERT OR REPLACE INTO papers
                (item_key, title, year, journal, doi, item_type, authors, tags, attachment_keys)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item.key, item.title, item.year, item.journal, item.doi, item.item_type,
            json.dumps(item.authors), json.dumps(item.tags), json.dumps(attachment_keys),
        ))
        self._conn.commit()

    def get(self, item_key: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM papers WHERE item_key = ?", (item_key,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_attachment_key(self, item_key: str) -> str | None:
        """Return the first attachment key for download. Multi-PDF selection is a future improvement."""
        row = self._conn.execute(
            "SELECT attachment_keys FROM papers WHERE item_key = ?", (item_key,)
        ).fetchone()
        if not row:
            return None
        keys = json.loads(row["attachment_keys"])
        return keys[0] if keys else None

    def delete(self, item_key: str) -> None:
        self._conn.execute("DELETE FROM papers WHERE item_key = ?", (item_key,))
        self._conn.commit()

    def title_search(self, query: str) -> list[str]:
        """Return item keys whose title contains any word from the query."""
        words = [w for w in query.lower().split() if len(w) > 1]
        if not words:
            return []
        conditions = " OR ".join("LOWER(title) LIKE ?" for _ in words)
        rows = self._conn.execute(
            f"SELECT item_key FROM papers WHERE {conditions}",
            [f"%{w}%" for w in words],
        ).fetchall()
        return [r["item_key"] for r in rows]

    def author_search(self, query: str) -> list[str]:
        """Return item keys with an author name matching the query."""
        rows = self._conn.execute(
            "SELECT item_key FROM papers WHERE LOWER(authors) LIKE ?",
            (f"%{query.lower()}%",),
        ).fetchall()
        return [r["item_key"] for r in rows]

    def tag_search(self, query: str) -> list[str]:
        """Return item keys where any tag matches any word in the query."""
        words = [w for w in query.lower().split() if len(w) > 1]
        if not words:
            return []
        conditions = " OR ".join("LOWER(tags) LIKE ?" for _ in words)
        rows = self._conn.execute(
            f"SELECT item_key FROM papers WHERE {conditions}",
            [f"%{w}%" for w in words],
        ).fetchall()
        return [r["item_key"] for r in rows]

    def filter(
        self,
        year: str | None = None,
        tags: list[str] | None = None,
        authors: list[str] | None = None,
    ) -> list[str]:
        """Return item keys matching all provided criteria."""
        conditions: list[str] = []
        params: list = []
        if year:
            conditions.append("year = ?")
            params.append(year)
        if tags:
            tag_conds = " OR ".join('LOWER(tags) LIKE ?' for _ in tags)
            conditions.append(f"({tag_conds})")
            params.extend(f'%"{t.lower()}"%' for t in tags)
        if authors:
            author_conds = " OR ".join("LOWER(authors) LIKE ?" for _ in authors)
            conditions.append(f"({author_conds})")
            params.extend(f"%{a.lower()}%" for a in authors)
        where = " AND ".join(conditions) if conditions else "1"
        rows = self._conn.execute(
            f"SELECT item_key FROM papers WHERE {where}", params
        ).fetchall()
        return [r["item_key"] for r in rows]

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        return {
            "title": row["title"],
            "authors": json.loads(row["authors"]),
            "year": row["year"],
            "journal": row["journal"],
            "doi": row["doi"],
            "tags": json.loads(row["tags"]),
            "item_type": row["item_type"],
            "attachment_keys": json.loads(row["attachment_keys"]),
        }
