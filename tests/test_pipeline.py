from unittest.mock import MagicMock, patch

import pytest

from zori.ingestion.pipeline import IngestionPipeline
from zori.ingestion.zotero import ZoteroAttachment, ZoteroItem
from zori.retrieval.vectorstore import MetadataStore


def _make_item(key="P1", version=1, with_pdf=True):
    attachments = [ZoteroAttachment(key="A1", filename="paper.pdf", parent_key=key, md5="abc")] if with_pdf else []
    return ZoteroItem(
        key=key,
        title="Test Paper",
        version=version,
        authors=["Alice"],
        year="2023",
        attachments=attachments,
    )


@pytest.fixture
def mock_zotero():
    zot = MagicMock()
    zot.fetch_all_items.return_value = [_make_item()]
    zot.download_pdf.return_value = b"fake-pdf-bytes"
    zot.get_library_version.return_value = 100
    return zot


@pytest.fixture
def mock_vector_store():
    vs = MagicMock()
    return vs


@pytest.fixture
def metadata_store(tmp_path):
    return MetadataStore(path=tmp_path / "zori.db")


@pytest.fixture
def pipeline(mock_zotero, mock_vector_store, metadata_store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("zori.ingestion.pipeline.PDFParser") as MockParser:
        mock_parser = MagicMock()
        mock_parser.extract_text.return_value = "Full paper text here."
        mock_parser.chunk.return_value = [MagicMock(item_key="P1", chunk_index=0)]
        MockParser.return_value = mock_parser
        p = IngestionPipeline(mock_zotero, mock_vector_store, metadata_store)
    return p


# --- run_full ---

def test_run_full_ingests_item(pipeline, mock_zotero, mock_vector_store, metadata_store):
    result = pipeline.run_full()

    assert result.ingested == 1
    assert result.failed == 0
    mock_vector_store.add_chunks.assert_called_once()
    assert metadata_store.get("P1") is not None


def test_run_full_skips_item_without_pdf(pipeline, mock_zotero):
    mock_zotero.fetch_all_items.return_value = [_make_item(with_pdf=False)]
    result = pipeline.run_full()
    assert result.skipped_no_pdf == 1
    assert result.ingested == 0


def test_run_full_skips_already_ingested(pipeline, mock_zotero):
    pipeline.run_full()
    result = pipeline.run_full()
    assert result.skipped_already_done == 1


def test_run_full_reingests_on_version_change(pipeline, mock_zotero, mock_vector_store):
    pipeline.run_full()
    mock_zotero.fetch_all_items.return_value = [_make_item(version=2)]
    result = pipeline.run_full()
    assert result.ingested == 1
    assert mock_vector_store.delete_item.called


def test_run_full_handles_pdf_error(pipeline, mock_zotero):
    mock_zotero.download_pdf.side_effect = Exception("network error")
    result = pipeline.run_full()
    assert result.failed == 1
    assert "P1" in result.failed_keys


def test_run_full_saves_library_version(pipeline, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pipeline.run_full()
    assert pipeline._state.get("library_version") == 100


# --- run_sync ---

def test_run_sync_uses_library_version(pipeline, mock_zotero):
    pipeline._state["library_version"] = 50
    pipeline.run_sync()
    mock_zotero.fetch_new_items.assert_called_once_with(since_version=50)


def test_run_sync_nothing_new(pipeline, mock_zotero):
    mock_zotero.fetch_new_items.return_value = []
    result = pipeline.run_sync()
    assert result.ingested == 0
