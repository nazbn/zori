import pytest

from zori.agents.paper_finder import SearchPlan


# ---------------------------------------------------------------------------
# SearchPlan Pydantic validators
# ---------------------------------------------------------------------------

def test_empty_string_fields_become_none():
    plan = SearchPlan(display_query="q", title="", author="", year_from="", year_to="", semantic_query="")
    assert plan.title is None
    assert plan.author is None
    assert plan.year_from is None
    assert plan.year_to is None
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
        year_from="2017",
        year_to="2017",
        tags=["NLP", "transformers"],
        lexical_queries=["attention", "self-attention"],
        semantic_query="transformer architecture",
    )
    assert plan.title == "Attention is All You Need"
    assert plan.author == "Vaswani"
    assert plan.year_from == "2017"
    assert plan.year_to == "2017"
    assert plan.tags == ["NLP", "transformers"]
    assert plan.lexical_queries == ["attention", "self-attention"]
    assert plan.semantic_query == "transformer architecture"


def test_mixed_empty_and_non_empty_fields():
    plan = SearchPlan(display_query="q", author="Smith", title="", tags=[], year_from="2020", year_to="2023")
    assert plan.author == "Smith"
    assert plan.year_from == "2020"
    assert plan.year_to == "2023"
    assert plan.title is None
    assert plan.tags is None


def test_display_query_always_required():
    with pytest.raises(Exception):
        SearchPlan()


# ---------------------------------------------------------------------------
# paper_finder_node — interrupt-based confirmation
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage
from zori.retrieval.search import SearchResult


def _make_result(key):
    return SearchResult(item_key=key, title=key, authors=[], year=None, journal=None, text="", score=0.9)


def _make_node_with_results(results):
    from zori.agents.paper_finder import make_paper_finder_node
    svc = MagicMock()
    svc.hybrid_search.return_value = results
    llm = MagicMock()
    from zori.agents.paper_finder import SearchPlan
    analyzer = MagicMock()
    analyzer.invoke.return_value = SearchPlan(display_query="test", semantic_query="test")
    llm.with_structured_output.return_value = analyzer
    return make_paper_finder_node(svc, llm)


def _state(query="test", intent="summarize"):
    return {"query": query, "intent": intent, "messages": [HumanMessage(content=query)], "search_results": []}


def test_no_results_returns_error_without_interrupt():
    node = _make_node_with_results([])
    with patch("zori.agents.paper_finder.interrupt") as mock_interrupt:
        result = node(_state())
    mock_interrupt.assert_not_called()
    assert result.get("target_key") is None
    assert result["response"] is not None


def test_single_result_yes_sets_target_key():
    node = _make_node_with_results([_make_result("P1")])
    with patch("zori.agents.paper_finder.interrupt", return_value="yes"):
        result = node(_state())
    assert result["target_key"] == "P1"


def test_single_result_no_returns_response():
    node = _make_node_with_results([_make_result("P1")])
    with patch("zori.agents.paper_finder.interrupt", return_value="no"):
        result = node(_state())
    assert result.get("target_key") is None
    assert result["response"] is not None


def test_multiple_results_numbered_reply_selects_correct():
    results = [_make_result("P1"), _make_result("P2"), _make_result("P3")]
    node = _make_node_with_results(results)
    with patch("zori.agents.paper_finder.interrupt", return_value="2"):
        result = node(_state())
    assert result["target_key"] == "P2"


def test_multiple_results_out_of_bounds_cancels():
    results = [_make_result("P1"), _make_result("P2")]
    node = _make_node_with_results(results)
    with patch("zori.agents.paper_finder.interrupt", return_value="9"):
        result = node(_state())
    assert result.get("target_key") is None


def test_multiple_results_invalid_reply_cancels():
    results = [_make_result("P1"), _make_result("P2")]
    node = _make_node_with_results(results)
    with patch("zori.agents.paper_finder.interrupt", return_value="maybe"):
        result = node(_state())
    assert result.get("target_key") is None
    assert result["response"] is not None
