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
# paper_finder_node — summarize mode returns candidates, never interrupts
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage
from langgraph.graph import END
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


def test_no_results_returns_error_and_empty_candidates():
    node = _make_node_with_results([])
    with patch("zori.agents.paper_finder.interrupt") as mock_interrupt:
        result = node(_state())
    mock_interrupt.assert_not_called()
    assert result["candidates"] == []
    assert result["response"] is not None


def test_results_returned_as_candidates_without_interrupt():
    node = _make_node_with_results([_make_result("P1"), _make_result("P2")])
    with patch("zori.agents.paper_finder.interrupt") as mock_interrupt:
        result = node(_state())
    mock_interrupt.assert_not_called()
    assert [r.item_key for r in result["candidates"]] == ["P1", "P2"]
    assert result["display_query"] == "test"
    assert "target_key" not in result


# ---------------------------------------------------------------------------
# confirm_paper_node — asks which candidate to summarize
# ---------------------------------------------------------------------------

def _confirm_state(keys):
    return {"candidates": [_make_result(k) for k in keys], "display_query": "test"}


def _confirm(keys, reply):
    """Run confirm_paper_node with a mocked reply; returns (Command, mock_interrupt)."""
    from zori.agents.paper_finder import confirm_paper_node
    with patch("zori.agents.paper_finder.interrupt", return_value=reply) as mock_interrupt:
        cmd = confirm_paper_node(_confirm_state(keys))
    return cmd, mock_interrupt


def test_single_result_yes_goes_to_summarization():
    cmd, mock_interrupt = _confirm(["P1"], "yes")
    assert cmd.goto == "summarization"
    assert cmd.update["target_key"] == "P1"
    assert mock_interrupt.call_args[0][0]["zotero_key"] == "P1"


def test_single_result_no_ends():
    cmd, _ = _confirm(["P1"], "no")
    assert cmd.goto == END
    assert "target_key" not in cmd.update
    assert cmd.update["response"]


def test_single_result_other_reply_redirects_to_router():
    cmd, _ = _confirm(["P1"], "not this one, the paper named BERT")
    assert cmd.goto == "router"
    assert cmd.update["query"] == "not this one, the paper named BERT"


def test_multiple_results_numbered_reply_selects_correct():
    cmd, _ = _confirm(["P1", "P2", "P3"], "2")
    assert cmd.goto == "summarization"
    assert cmd.update["target_key"] == "P2"
    assert [r.item_key for r in cmd.update["search_results"]] == ["P1", "P2", "P3"]


def test_multiple_results_out_of_bounds_cancels():
    cmd, _ = _confirm(["P1", "P2"], "9")
    assert cmd.goto == END
    assert "no option 9" in cmd.update["response"]


def test_multiple_results_only_first_three_offered():
    cmd, mock_interrupt = _confirm(["P1", "P2", "P3", "P4"], "4")
    assert cmd.goto == END
    assert "P4" not in mock_interrupt.call_args[0][0]["text"]


def test_multiple_results_none_cancels():
    cmd, _ = _confirm(["P1", "P2"], "none")
    assert cmd.goto == END
    assert cmd.update["response"]


def test_multiple_results_other_reply_redirects_to_router():
    cmd, _ = _confirm(["P1", "P2"], "maybe the older one?")
    assert cmd.goto == "router"
    question_msg, reply_msg = cmd.update["messages"]
    assert "Which one?" in question_msg.content
    assert reply_msg.content == "maybe the older one?"
