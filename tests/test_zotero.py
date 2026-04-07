import pytest
from zori.ingestion.zotero import ZoteroClient, ZoteroItem, ZoteroAttachment


# --- Fixtures ---

def _make_raw_item(key="ABC123", title="Test Paper", version=42,
                   item_type="journalArticle", creators=None, date="2023-05-01",
                   abstract="An abstract.", tags=None, doi="10.1234/test"):
    return {
        "key": key,
        "version": version,
        "data": {
            "key": key,
            "itemType": item_type,
            "title": title,
            "creators": creators or [
                {"creatorType": "author", "firstName": "Jane", "lastName": "Doe"},
                {"creatorType": "author", "firstName": "John", "lastName": "Smith"},
            ],
            "date": date,
            "abstractNote": abstract,
            "publicationTitle": "Journal of Testing",
            "DOI": doi,
            "tags": [{"tag": t} for t in (tags or ["ml", "rag"])],
        },
    }


def _make_raw_attachment(key="ATT001", parent_key="ABC123",
                         filename="paper.pdf", md5="abc123"):
    return {
        "key": key,
        "version": 10,
        "data": {
            "key": key,
            "itemType": "attachment",
            "parentItem": parent_key,
            "linkMode": "imported_url",
            "contentType": "application/pdf",
            "filename": filename,
            "md5": md5,
        },
    }


# --- _parse_paper ---

def test_parse_paper_basic(monkeypatch):
    client = _make_client()
    raw = _make_raw_item()
    item = client._parse_paper(raw)

    assert item.key == "ABC123"
    assert item.title == "Test Paper"
    assert item.version == 42
    assert item.authors == ["Jane Doe", "John Smith"]
    assert item.year == "2023"
    assert item.abstract == "An abstract."
    assert item.journal == "Journal of Testing"
    assert item.doi == "10.1234/test"
    assert item.tags == ["ml", "rag"]
    assert item.item_type == "journalArticle"


def test_parse_paper_empty_abstract():
    client = _make_client()
    raw = _make_raw_item(abstract="")
    item = client._parse_paper(raw)
    assert item.abstract is None


def test_parse_paper_no_date():
    client = _make_client()
    raw = _make_raw_item(date="")
    item = client._parse_paper(raw)
    assert item.year is None


def test_parse_paper_single_name_author():
    client = _make_client()
    raw = _make_raw_item(creators=[{"creatorType": "author", "name": "Plato"}])
    item = client._parse_paper(raw)
    assert item.authors == ["Plato"]


# --- _parse_items ---

def test_parse_items_links_attachment_to_paper():
    client = _make_client()
    raw_paper = _make_raw_item(key="P1")
    raw_att = _make_raw_attachment(key="A1", parent_key="P1")
    items = client._parse_items([raw_paper, raw_att])

    assert len(items) == 1
    assert items[0].key == "P1"
    assert len(items[0].attachments) == 1
    assert items[0].attachments[0].key == "A1"


def test_parse_items_skips_non_pdf_attachments():
    client = _make_client()
    raw_paper = _make_raw_item(key="P1")
    html_att = _make_raw_attachment(key="H1", parent_key="P1")
    html_att["data"]["contentType"] = "text/html"
    items = client._parse_items([raw_paper, html_att])

    assert len(items[0].attachments) == 0


def test_parse_items_top_level_attachment():
    client = _make_client()
    raw_att = _make_raw_attachment(key="A1", parent_key=None)
    del raw_att["data"]["parentItem"]
    items = client._parse_items([raw_att])

    assert len(items) == 1
    assert items[0].key == "A1"
    assert items[0].item_type == "attachment"


def test_parse_items_skips_notes_and_annotations():
    client = _make_client()
    note = {"key": "N1", "version": 1, "data": {"key": "N1", "itemType": "note"}}
    annotation = {"key": "AN1", "version": 1, "data": {"key": "AN1", "itemType": "annotation"}}
    items = client._parse_items([note, annotation])
    assert items == []


def test_parse_items_multiple_attachments():
    client = _make_client()
    raw_paper = _make_raw_item(key="P1")
    att1 = _make_raw_attachment(key="A1", parent_key="P1", filename="v1.pdf")
    att2 = _make_raw_attachment(key="A2", parent_key="P1", filename="v2.pdf")
    items = client._parse_items([raw_paper, att1, att2])

    assert len(items[0].attachments) == 2


# --- helpers ---

def _make_client():
    """Create a ZoteroClient without making any API calls."""
    from unittest.mock import MagicMock
    from pyzotero import zotero
    client = object.__new__(ZoteroClient)
    client._zot = MagicMock()
    return client
