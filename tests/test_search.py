from unittest.mock import MagicMock, call

import pytest

from zori.retrieval.search import SearchService, _rrf_combine
from zori.retrieval.vector import ChunkResult


# ---------------------------------------------------------------------------
# _rrf_combine
# ---------------------------------------------------------------------------

def test_rrf_combine_higher_weight_ranks_first():
    # A is in weight-3 list, B is in weight-1 list, both at rank 0
    result = _rrf_combine([["A"], ["B"]], [3.0, 1.0])
    assert result[0] == "A"


def test_rrf_combine_item_in_multiple_lists_scores_higher():
    # A appears in both lists; B and C each appear in only one
    result = _rrf_combine([["A", "B"], ["A", "C"]], [1.0, 1.0])
    assert result[0] == "A"


def test_rrf_combine_empty_returns_empty():
    assert _rrf_combine([], []) == []


def test_rrf_combine_single_list_preserves_order():
    result = _rrf_combine([["X", "Y", "Z"]], [1.0])
    assert result == ["X", "Y", "Z"]


def test_rrf_combine_weight_zero_list_ignored():
    # weight=0 contributes nothing; only the weight-1 list should determine order
    result = _rrf_combine([["A", "B"], ["B", "A"]], [0.0, 1.0])
    assert result[0] == "B"


def test_rrf_combine_custom_k():
    # k=0: score = weight / (0 + rank + 1). Order should still be preserved.
    result = _rrf_combine([["A", "B"]], [1.0], k=0)
    assert result == ["A", "B"]


# ---------------------------------------------------------------------------
# hybrid_search — fixtures
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
# hybrid_search — no inputs
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
# hybrid_search — author / year hard filters
# ---------------------------------------------------------------------------

def test_hybrid_search_author_filter_excludes_non_matching(mock_stores):
    vector_store, metadata_store, lexical_index = mock_stores
    metadata_store.filter.return_value = ["P2"]
    metadata_store.get.side_effect = _meta
    lexical_index.search_papers.return_value = [("P1", -1.0), ("P2", -0.5)]
    lexical_index.search_chunks.return_value = []

    svc = SearchService(vector_store, metadata_store, lexical_index)
    results = svc.hybrid_search(lexical_queries=["neural"], author="Smith")
    keys = [r.item_key for r in results]
    assert "P1" not in keys
    assert "P2" in keys


def test_hybrid_search_year_filter_excludes_non_matching(mock_stores):
    vector_store, metadata_store, lexical_index = mock_stores
    metadata_store.filter.return_value = ["P1"]
    metadata_store.get.side_effect = _meta
    lexical_index.search_papers.return_value = [("P1", -1.0), ("P2", -0.5)]
    lexical_index.search_chunks.return_value = []

    svc = SearchService(vector_store, metadata_store, lexical_index)
    results = svc.hybrid_search(lexical_queries=["neural"], year="2023")
    keys = [r.item_key for r in results]
    assert "P1" in keys
    assert "P2" not in keys


# ---------------------------------------------------------------------------
# hybrid_search — multiple lexical queries
# ---------------------------------------------------------------------------

def test_hybrid_search_multiple_lexical_queries_merged(mock_stores):
    vector_store, metadata_store, lexical_index = mock_stores
    metadata_store.get.side_effect = _meta
    # First query matches P1, second matches P2
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
    # P1 appears in both lexical query results; P2 only in one
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
# hybrid_search — tags as soft scoring (not hard filter)
# ---------------------------------------------------------------------------

def test_hybrid_search_tags_do_not_exclude_non_tagged_papers(mock_stores):
    vector_store, metadata_store, lexical_index = mock_stores
    metadata_store.get.side_effect = _meta
    # P1 matches lexical but has no tag; P2 matches tag search
    lexical_index.search_papers.return_value = [("P1", -1.0)]
    lexical_index.search_chunks.return_value = []
    lexical_index.search_tags.return_value = [("P2", -1.0)]

    svc = SearchService(vector_store, metadata_store, lexical_index)
    results = svc.hybrid_search(lexical_queries=["neural"], tags=["ml"])
    keys = [r.item_key for r in results]
    assert "P1" in keys  # present despite not matching tag
    assert "P2" in keys  # present because of tag match


def test_hybrid_search_tags_do_not_trigger_metadata_filter(mock_stores):
    vector_store, metadata_store, lexical_index = mock_stores
    lexical_index.search_tags.return_value = [("P1", -1.0)]
    lexical_index.search_papers.return_value = []
    lexical_index.search_chunks.return_value = []
    metadata_store.get.side_effect = _meta

    svc = SearchService(vector_store, metadata_store, lexical_index)
    svc.hybrid_search(tags=["ml"])
    # filter() should NOT be called when only tags are provided (no author/year)
    metadata_store.filter.assert_not_called()
