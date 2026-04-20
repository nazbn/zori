from unittest.mock import MagicMock

import pytest

from zori.retrieval.search import SearchService


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_stores():
    vector_store = MagicMock()
    metadata_store = MagicMock()
    lexical_index = MagicMock()

    vector_store.search.return_value = []
    metadata_store.filter.return_value = []
    metadata_store.get.return_value = {"title": "Unknown", "authors": [], "year": None, "journal": None}
    lexical_index.search_papers.return_value = []
    lexical_index.search_chunks.return_value = []
    lexical_index.search_tags.return_value = []
    lexical_index.search_title.return_value = []

    return vector_store, metadata_store, lexical_index


def _meta(key):
    return {"title": key, "authors": [], "year": None, "journal": None}


# ---------------------------------------------------------------------------
# no inputs
# ---------------------------------------------------------------------------

def test_hybrid_search_no_inputs_returns_empty(mock_stores):
    vector_store, metadata_store, lexical_index = mock_stores
    svc = SearchService(vector_store, metadata_store, lexical_index)
    assert svc.hybrid_search() == []


def test_hybrid_search_all_retrievers_empty_returns_empty(mock_stores):
    vector_store, metadata_store, lexical_index = mock_stores
    svc = SearchService(vector_store, metadata_store, lexical_index)
    results = svc.hybrid_search(lexical_queries=["topic"], semantic_query="topic", tags=["ml"])
    assert results == []


# ---------------------------------------------------------------------------
# author / year — soft scoring (not hard filters)
# ---------------------------------------------------------------------------

def test_hybrid_search_author_boosts_matching_papers(mock_stores):
    vector_store, metadata_store, lexical_index = mock_stores
    metadata_store.filter.return_value = ["P2"]  # P2 matches author
    metadata_store.get.side_effect = _meta
    lexical_index.search_papers.return_value = [("P1", -1.0), ("P2", -0.5)]
    lexical_index.search_chunks.return_value = []

    svc = SearchService(vector_store, metadata_store, lexical_index)
    results = svc.hybrid_search(lexical_queries=["neural"], author="Smith")
    keys = [r.item_key for r in results]
    assert "P1" in keys                            # still present (soft scoring)
    assert "P2" in keys
    assert keys.index("P2") < keys.index("P1")    # P2 ranks higher due to author match


def test_hybrid_search_year_boosts_matching_papers(mock_stores):
    vector_store, metadata_store, lexical_index = mock_stores
    metadata_store.filter.return_value = ["P1"]  # P1 matches year
    metadata_store.get.side_effect = _meta
    lexical_index.search_papers.return_value = [("P2", -1.0), ("P1", -0.5)]
    lexical_index.search_chunks.return_value = []

    svc = SearchService(vector_store, metadata_store, lexical_index)
    results = svc.hybrid_search(lexical_queries=["neural"], year="2023")
    keys = [r.item_key for r in results]
    assert "P1" in keys
    assert "P2" in keys
    assert keys.index("P1") < keys.index("P2")    # P1 ranks higher due to year match


def test_hybrid_search_author_only_returns_matching_papers(mock_stores):
    vector_store, metadata_store, lexical_index = mock_stores
    metadata_store.filter.return_value = ["P1", "P2"]
    metadata_store.get.side_effect = _meta

    svc = SearchService(vector_store, metadata_store, lexical_index)
    results = svc.hybrid_search(author="Vaswani")
    keys = [r.item_key for r in results]
    assert "P1" in keys
    assert "P2" in keys


# ---------------------------------------------------------------------------
# multiple lexical queries
# ---------------------------------------------------------------------------

def test_hybrid_search_multiple_lexical_queries_merged(mock_stores):
    vector_store, metadata_store, lexical_index = mock_stores
    metadata_store.get.side_effect = _meta
    lexical_index.search_papers.side_effect = [
        [("P1", -1.0)],
        [("P2", -1.0)],
    ]
    lexical_index.search_chunks.return_value = []

    svc = SearchService(vector_store, metadata_store, lexical_index)
    results = svc.hybrid_search(lexical_queries=["swin2sr", "swin transformer super resolution"])
    keys = [r.item_key for r in results]
    assert "P1" in keys
    assert "P2" in keys


def test_hybrid_search_multiple_lexical_queries_calls_search_per_query(mock_stores):
    vector_store, metadata_store, lexical_index = mock_stores
    lexical_index.search_papers.return_value = []
    lexical_index.search_chunks.return_value = []

    svc = SearchService(vector_store, metadata_store, lexical_index)
    svc.hybrid_search(lexical_queries=["q1", "q2", "q3"])
    assert lexical_index.search_papers.call_count == 3
    assert lexical_index.search_chunks.call_count == 3


def test_hybrid_search_paper_in_both_lexical_queries_ranks_high(mock_stores):
    vector_store, metadata_store, lexical_index = mock_stores
    metadata_store.get.side_effect = _meta
    lexical_index.search_papers.side_effect = [
        [("P1", -1.0), ("P2", -0.5)],
        [("P1", -1.0), ("P3", -0.5)],
    ]
    lexical_index.search_chunks.return_value = []

    svc = SearchService(vector_store, metadata_store, lexical_index)
    results = svc.hybrid_search(lexical_queries=["q1", "q2"])
    keys = [r.item_key for r in results]
    assert keys[0] == "P1"  # P1 matched both queries, should rank first


# ---------------------------------------------------------------------------
# tags as soft scoring
# ---------------------------------------------------------------------------

def test_hybrid_search_tags_do_not_exclude_non_tagged_papers(mock_stores):
    vector_store, metadata_store, lexical_index = mock_stores
    metadata_store.get.side_effect = _meta
    lexical_index.search_papers.return_value = [("P1", -1.0)]
    lexical_index.search_chunks.return_value = []
    lexical_index.search_tags.return_value = [("P2", -1.0)]

    svc = SearchService(vector_store, metadata_store, lexical_index)
    results = svc.hybrid_search(lexical_queries=["neural"], tags=["ml"])
    keys = [r.item_key for r in results]
    assert "P1" in keys
    assert "P2" in keys


def test_hybrid_search_tags_do_not_trigger_metadata_filter(mock_stores):
    vector_store, metadata_store, lexical_index = mock_stores
    lexical_index.search_tags.return_value = [("P1", -1.0)]
    lexical_index.search_papers.return_value = []
    lexical_index.search_chunks.return_value = []
    metadata_store.get.side_effect = _meta

    svc = SearchService(vector_store, metadata_store, lexical_index)
    svc.hybrid_search(tags=["ml"])
    metadata_store.filter.assert_not_called()
