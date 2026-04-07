import pytest

from zori.agents.paper_finder import SearchPlan


# ---------------------------------------------------------------------------
# SearchPlan Pydantic validators
# ---------------------------------------------------------------------------

def test_empty_string_fields_become_none():
    plan = SearchPlan(display_query="q", title="", author="", year="", semantic_query="")
    assert plan.title is None
    assert plan.author is None
    assert plan.year is None
    assert plan.semantic_query is None


def test_empty_list_fields_become_none():
    plan = SearchPlan(display_query="q", tags=[], lexical_queries=[])
    assert plan.tags is None
    assert plan.lexical_queries is None


def test_non_empty_values_preserved():
    plan = SearchPlan(
        display_query="transformers",
        title="Attention is All You Need",
        author="Vaswani",
        year="2017",
        tags=["NLP", "transformers"],
        lexical_queries=["attention", "self-attention"],
        semantic_query="transformer architecture",
    )
    assert plan.title == "Attention is All You Need"
    assert plan.author == "Vaswani"
    assert plan.year == "2017"
    assert plan.tags == ["NLP", "transformers"]
    assert plan.lexical_queries == ["attention", "self-attention"]
    assert plan.semantic_query == "transformer architecture"


def test_mixed_empty_and_non_empty_fields():
    plan = SearchPlan(display_query="q", author="Smith", title="", tags=[], year="2023")
    assert plan.author == "Smith"
    assert plan.year == "2023"
    assert plan.title is None
    assert plan.tags is None


def test_display_query_always_required():
    with pytest.raises(Exception):
        SearchPlan()


# ---------------------------------------------------------------------------
# _handle_confirmation — numbered replies and edge cases
# ---------------------------------------------------------------------------

from langchain_core.messages import HumanMessage
from zori.agents.paper_finder import _handle_confirmation
from zori.retrieval.search import SearchResult


def _make_result(key):
    return SearchResult(item_key=key, title=key, authors=[], year=None, journal=None, text="", score=0.9)


def test_confirmation_numbered_reply_selects_correct_result():
    state = {
        "query": "2",
        "search_results": [_make_result("P1"), _make_result("P2"), _make_result("P3")],
    }
    result = _handle_confirmation(state)
    assert result["target_key"] == "P2"
    assert result["pending_confirmation"] is False


def test_confirmation_numbered_reply_first_result():
    state = {"query": "1", "search_results": [_make_result("P1"), _make_result("P2")]}
    result = _handle_confirmation(state)
    assert result["target_key"] == "P1"


def test_confirmation_numbered_reply_out_of_bounds_cancels():
    state = {"query": "5", "search_results": [_make_result("P1"), _make_result("P2")]}
    result = _handle_confirmation(state)
    assert result["target_key"] is None
    assert result["pending_confirmation"] is False


def test_confirmation_numbered_reply_zero_cancels():
    state = {"query": "0", "search_results": [_make_result("P1")]}
    result = _handle_confirmation(state)
    assert result["target_key"] is None


def test_confirmation_yes_sets_candidate_key():
    state = {"query": "yes", "search_results": [], "candidate_key": "P1"}
    result = _handle_confirmation(state)
    assert result["target_key"] == "P1"
    assert result["pending_confirmation"] is False


def test_confirmation_y_shorthand():
    state = {"query": "y", "search_results": [], "candidate_key": "P1"}
    result = _handle_confirmation(state)
    assert result["target_key"] == "P1"


def test_confirmation_invalid_reply_cancels():
    state = {"query": "maybe", "search_results": [], "candidate_key": "P1"}
    result = _handle_confirmation(state)
    assert result["target_key"] is None
    assert result["pending_confirmation"] is False
    assert result["response"] is not None


def test_confirmation_reply_with_whitespace_stripped():
    state = {"query": "  yes  ", "search_results": [], "candidate_key": "P1"}
    result = _handle_confirmation(state)
    assert result["target_key"] == "P1"
