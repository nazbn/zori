import pytest

from zori.ingestion.zotero import ZoteroAttachment, ZoteroItem
from zori.retrieval.vectorstore import MetadataStore


@pytest.fixture
def store(tmp_path):
    return MetadataStore(path=tmp_path / "metadata.json")


def _make_item(key="P1", title="Test Paper", year="2023",
               tags=None, authors=None, attachments=None):
    return ZoteroItem(
        key=key,
        title=title,
        version=1,
        authors=authors or ["Alice", "Bob"],
        year=year,
        journal="Test Journal",
        doi="10.1234/test",
        tags=tags or ["ml"],
        attachments=attachments or [
            ZoteroAttachment(key="A1", filename="paper.pdf", parent_key=key, md5="abc")
        ],
    )


# --- save / get ---

def test_save_and_get(store):
    item = _make_item()
    store.save(item)
    meta = store.get("P1")

    assert meta["title"] == "Test Paper"
    assert meta["authors"] == ["Alice", "Bob"]
    assert meta["year"] == "2023"
    assert meta["attachment_keys"] == ["A1"]


def test_get_missing_returns_none(store):
    assert store.get("MISSING") is None


def test_get_attachment_key(store):
    store.save(_make_item())
    assert store.get_attachment_key("P1") == "A1"


def test_get_attachment_key_missing(store):
    assert store.get_attachment_key("MISSING") is None


def test_get_attachment_key_no_attachments(store):
    item = _make_item(attachments=[])
    store.save(item)
    assert store.get_attachment_key("P1") is None


# --- delete ---

def test_delete(store):
    store.save(_make_item())
    store.delete("P1")
    assert store.get("P1") is None


def test_delete_nonexistent_does_not_raise(store):
    store.delete("MISSING")  # should not raise


# --- filter ---

def test_filter_by_year(store):
    store.save(_make_item(key="P1", year="2023"))
    store.save(_make_item(key="P2", year="2024"))
    results = store.filter(year="2023")
    assert results == ["P1"]


def test_filter_by_tag(store):
    store.save(_make_item(key="P1", tags=["rag", "llm"]))
    store.save(_make_item(key="P2", tags=["cv"]))
    results = store.filter(tags=["rag"])
    assert "P1" in results
    assert "P2" not in results


def test_filter_by_author(store):
    store.save(_make_item(key="P1", authors=["Alice Smith", "Bob Jones"]))
    store.save(_make_item(key="P2", authors=["Carol White"]))
    results = store.filter(authors=["alice"])
    assert "P1" in results
    assert "P2" not in results


def test_filter_combined(store):
    store.save(_make_item(key="P1", year="2023", tags=["rag"]))
    store.save(_make_item(key="P2", year="2023", tags=["cv"]))
    results = store.filter(year="2023", tags=["rag"])
    assert results == ["P1"]


# --- persistence ---

def test_persists_to_disk(tmp_path):
    store1 = MetadataStore(path=tmp_path / "metadata.json")
    store1.save(_make_item())

    store2 = MetadataStore(path=tmp_path / "metadata.json")
    assert store2.get("P1") is not None
