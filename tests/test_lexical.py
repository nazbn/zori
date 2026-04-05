from pathlib import Path

import pytest

from zori.ingestion.pdf import TextChunk
from zori.ingestion.zotero import ZoteroItem, ZoteroAttachment
from zori.retrieval.lexical import LexicalIndex


@pytest.fixture
def index(tmp_path):
    return LexicalIndex(path=tmp_path / "zori.db")


@pytest.fixture
def paper():
    return ZoteroItem(
        key="ABC123",
        title="DEM Super-Resolution with Generative Adversarial Networks",
        abstract="We propose a GAN-based model for digital elevation model super-resolution.",
        tags=["DEM", "super-resolution", "GAN"],
        authors=["Alice Smith"],
        year="2021",
        attachments=[ZoteroAttachment(key="ATT1", filename="paper.pdf", parent_key="ABC123")],
    )


@pytest.fixture
def chunks():
    return [
        TextChunk(item_key="ABC123", chunk_index=0, text="Digital elevation models are used in terrain analysis."),
        TextChunk(item_key="ABC123", chunk_index=1, text="Our GAN achieves state-of-the-art super-resolution."),
    ]


def test_add_paper_inserts_row(index, paper):
    index.add_paper(paper)
    row = index._conn.execute("SELECT * FROM papers_fts WHERE item_key = 'ABC123'").fetchone()
    assert row is not None
    assert row["title"] == paper.title
    assert row["abstract"] == paper.abstract
    assert "DEM" in row["tags"]


def test_add_chunks_inserts_rows(index, chunks):
    index.add_chunks(chunks)
    rows = index._conn.execute("SELECT * FROM chunks_fts WHERE item_key = 'ABC123'").fetchall()
    assert len(rows) == 2
    texts = {r["chunk_text"] for r in rows}
    assert "Digital elevation models are used in terrain analysis." in texts


def test_delete_item_removes_all(index, paper, chunks):
    index.add_paper(paper)
    index.add_chunks(chunks)
    index.delete_item("ABC123")
    assert index._conn.execute("SELECT COUNT(*) FROM papers_fts WHERE item_key='ABC123'").fetchone()[0] == 0
    assert index._conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE item_key='ABC123'").fetchone()[0] == 0


def test_add_paper_none_abstract(index):
    item = ZoteroItem(key="X1", title="A paper without abstract", abstract=None, tags=[])
    index.add_paper(item)
    row = index._conn.execute("SELECT abstract FROM papers_fts WHERE item_key='X1'").fetchone()
    assert row["abstract"] == ""


def test_add_chunks_empty_list(index):
    index.add_chunks([])  # should not raise
    count = index._conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    assert count == 0


def test_search_papers_returns_ranked_results(index, paper, chunks):
    index.add_paper(paper)
    index.add_chunks(chunks)
    results = index.search_papers("GAN super-resolution")
    assert len(results) > 0
    keys = [r[0] for r in results]
    assert "ABC123" in keys
    # scores are negative floats (BM25 convention)
    assert all(isinstance(r[1], float) for r in results)


def test_search_chunks_returns_ranked_results(index, chunks):
    index.add_chunks(chunks)
    results = index.search_chunks("GAN super-resolution")
    assert len(results) > 0
    keys = [r[0] for r in results]
    assert "ABC123" in keys


def test_search_chunks_deduplicates_by_paper(index):
    chunks = [
        TextChunk(item_key="X1", chunk_index=0, text="neural network training"),
        TextChunk(item_key="X1", chunk_index=1, text="neural network inference"),
        TextChunk(item_key="X2", chunk_index=0, text="neural network architecture"),
    ]
    index.add_chunks(chunks)
    results = index.search_chunks("neural network")
    keys = [r[0] for r in results]
    assert keys.count("X1") == 1
    assert keys.count("X2") == 1


def test_search_title_finds_paper(index, paper):
    index.add_paper(paper)
    keys = index.search_title("generative adversarial")
    assert "ABC123" in keys


def test_search_title_no_match_returns_empty(index, paper):
    index.add_paper(paper)
    keys = index.search_title("completely unrelated topic xyz")
    assert keys == []


def test_search_papers_invalid_query_returns_empty(index, paper):
    index.add_paper(paper)
    # FTS5 special character that would normally cause a parse error
    results = index.search_papers('"')
    assert results == []


def test_tables_persist_across_connections(tmp_path, paper, chunks):
    path = tmp_path / "zori.db"
    idx1 = LexicalIndex(path=path)
    idx1.add_paper(paper)
    idx1.add_chunks(chunks)
    idx1.close()

    idx2 = LexicalIndex(path=path)
    row = idx2._conn.execute("SELECT title FROM papers_fts WHERE item_key='ABC123'").fetchone()
    assert row["title"] == paper.title
    chunk_count = idx2._conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE item_key='ABC123'").fetchone()[0]
    assert chunk_count == 2
