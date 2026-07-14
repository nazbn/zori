from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from zori.agents.summarization import SummaryOutput


def _fresh_state(**overrides) -> dict:
    base = {
        "messages": [],
        "query": "",
        "intent": "",
        "target_key": None,
        "candidate_key": None,
        "display_query": "",
        "search_results": [],
        "summary": None,
        "response": None,
    }
    base.update(overrides)
    return base


# --- Routing functions ---

from langgraph.graph import END
from zori.agents.graph import _route_from_paper_finder, _route_from_router


class TestRouting:
    def _state(self, **kw):
        return {"intent": "", "target_key": None, "search_results": [], **kw}

    def test_search_goes_to_paper_finder(self):
        assert _route_from_router(self._state(intent="search")) == "paper_finder"

    def test_summarize_without_key_goes_to_paper_finder(self):
        assert _route_from_router(self._state(intent="summarize")) == "paper_finder"

    def test_summarize_with_key_goes_to_summarization(self):
        assert _route_from_router(self._state(intent="summarize", target_key="K1")) == "summarization"

    def test_general_goes_to_end(self):
        assert _route_from_router(self._state(intent="general")) == END

    def test_paper_finder_with_target_and_summarize_goes_to_summarization(self):
        assert _route_from_paper_finder(self._state(intent="summarize", target_key="K1")) == "summarization"

    def test_paper_finder_without_target_goes_to_end(self):
        assert _route_from_paper_finder(self._state(intent="summarize")) == END

    def test_paper_finder_search_intent_goes_to_end(self):
        assert _route_from_paper_finder(self._state(intent="search", target_key="K1")) == END


# --- Router ---

class TestRouter:
    @pytest.fixture
    def router_node(self):
        from zori.agents.router import make_router_node
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        return make_router_node(mock_llm), mock_structured

    def test_search_intent(self, router_node):
        node, mock_llm = router_node
        from zori.agents.router import RouterOutput
        mock_llm.invoke.return_value = RouterOutput(intent="search")
        state = _fresh_state(
            messages=[HumanMessage(content="papers on transformers")],
            query="papers on transformers",
        )
        result = node(state)
        assert result["intent"] == "search"

    def test_summarize_intent_with_key(self, router_node):
        node, mock_llm = router_node
        from zori.agents.router import RouterOutput
        mock_llm.invoke.return_value = RouterOutput(intent="summarize", target_key="ABC123")
        state = _fresh_state(
            messages=[HumanMessage(content="summarize ABC123")],
            query="summarize ABC123",
        )
        result = node(state)
        assert result["intent"] == "summarize"
        assert result["target_key"] == "ABC123"

    def test_general_intent_returns_response(self, router_node):
        node, mock_llm = router_node
        from zori.agents.router import RouterOutput
        mock_llm.invoke.return_value = RouterOutput(
            intent="general", response="RAG stands for Retrieval-Augmented Generation."
        )
        state = _fresh_state(
            messages=[HumanMessage(content="what is RAG?")],
            query="what is RAG?",
        )
        result = node(state)
        assert result["intent"] == "general"
        assert "RAG" in result["response"]
        assert isinstance(result["messages"][0], AIMessage)


# --- PaperFinder ---

class TestPaperFinder:
    @pytest.fixture
    def search_service(self):
        from zori.retrieval.search import SearchResult
        svc = MagicMock()
        result = SearchResult(
            item_key="P1", title="Attention is All You Need",
            authors=["Vaswani"], year="2017", journal="NeurIPS",
            text="The Transformer model...", score=0.92,
        )
        svc.hybrid_search.return_value = [result]
        return svc

    @pytest.fixture
    def mock_llm(self):
        from zori.agents.paper_finder import SearchPlan
        llm = MagicMock()
        plan = SearchPlan(display_query="test", semantic_query="test")
        analyzer = MagicMock()
        analyzer.invoke.return_value = plan
        llm.with_structured_output.return_value = analyzer
        return llm

    def test_display_mode_returns_results(self, search_service, mock_llm):
        from zori.agents.paper_finder import make_paper_finder_node
        node = make_paper_finder_node(search_service, mock_llm)
        state = _fresh_state(query="transformers", intent="search",
                             messages=[HumanMessage(content="transformers")])
        result = node(state)
        assert len(result["search_results"]) == 1
        assert result["response"] == ""  # display layer handles formatting

    def test_find_for_summarize_interrupts_with_question(self, search_service, mock_llm):
        from unittest.mock import patch
        from zori.agents.paper_finder import make_paper_finder_node
        node = make_paper_finder_node(search_service, mock_llm)
        state = _fresh_state(query="attention paper", intent="summarize",
                             messages=[HumanMessage(content="attention paper")])
        with patch("zori.agents.paper_finder.interrupt", return_value="yes") as mock_interrupt:
            result = node(state)
        mock_interrupt.assert_called_once()
        assert result["target_key"] == "P1"

    def test_find_for_summarize_no_clears_target(self, search_service, mock_llm):
        from unittest.mock import patch
        from zori.agents.paper_finder import make_paper_finder_node
        node = make_paper_finder_node(search_service, mock_llm)
        state = _fresh_state(query="attention paper", intent="summarize",
                             messages=[HumanMessage(content="attention paper")])
        with patch("zori.agents.paper_finder.interrupt", return_value="no"):
            result = node(state)
        assert result.get("target_key") is None


# --- Summarization ---

class TestSummarization:
    @pytest.fixture
    def mock_components(self):
        metadata_store = MagicMock()
        lexical_index = MagicMock()
        zotero_client = MagicMock()
        llm = MagicMock()

        metadata_store.get.return_value = {
            "title": "Test Paper", "authors": ["Smith"], "year": "2023",
        }
        lexical_index.get_full_text.return_value = "This is the full paper text about neural networks."

        mock_output = SummaryOutput(
            overview="A great paper.",
            contributions=["Novel approach"],
            methods="Deep learning",
            findings="State-of-the-art results.",
        )
        structured_llm = MagicMock()
        structured_llm.invoke.return_value = mock_output
        llm.with_structured_output.return_value = structured_llm

        return metadata_store, lexical_index, zotero_client, llm

    def test_summarization_yes_saves_note(self, mock_components):
        from unittest.mock import patch
        metadata_store, lexical_index, zotero_client, llm = mock_components
        from zori.agents.summarization import make_summarization_node
        node = make_summarization_node(zotero_client, llm, metadata_store, lexical_index)
        state = _fresh_state(target_key="P1", messages=[HumanMessage(content="summarize this")])
        with patch("zori.agents.summarization.interrupt", return_value="yes"):
            result = node(state)
        zotero_client.write_note.assert_called_once()
        assert "saved" in result["response"].lower()

    def test_summarization_no_skips_save(self, mock_components):
        from unittest.mock import patch
        metadata_store, lexical_index, zotero_client, llm = mock_components
        from zori.agents.summarization import make_summarization_node
        node = make_summarization_node(zotero_client, llm, metadata_store, lexical_index)
        state = _fresh_state(target_key="P1", messages=[HumanMessage(content="summarize")])
        with patch("zori.agents.summarization.interrupt", return_value="no"):
            result = node(state)
        zotero_client.write_note.assert_not_called()
        assert result["response"] is not None

    def test_summarization_interrupt_contains_title_and_sections(self, mock_components):
        from unittest.mock import patch
        metadata_store, lexical_index, zotero_client, llm = mock_components
        from zori.agents.summarization import make_summarization_node
        node = make_summarization_node(zotero_client, llm, metadata_store, lexical_index)
        state = _fresh_state(target_key="P1", messages=[HumanMessage(content="summarize")])
        with patch("zori.agents.summarization.interrupt", return_value="no") as mock_interrupt:
            node(state)
        call_value = mock_interrupt.call_args[0][0]
        text = call_value.get("text", "") if isinstance(call_value, dict) else str(call_value)
        assert "Test Paper" in text
        assert "Overview" in text

    def test_summarization_write_failure_returns_error(self, mock_components):
        from unittest.mock import patch
        metadata_store, lexical_index, zotero_client, llm = mock_components
        zotero_client.write_note.side_effect = RuntimeError("API error")
        from zori.agents.summarization import make_summarization_node
        node = make_summarization_node(zotero_client, llm, metadata_store, lexical_index)
        state = _fresh_state(target_key="P1", messages=[HumanMessage(content="summarize")])
        with patch("zori.agents.summarization.interrupt", return_value="yes"):
            result = node(state)
        assert "failed" in result["response"].lower()

    def test_summarization_missing_metadata_returns_error(self, mock_components):
        metadata_store, lexical_index, zotero_client, llm = mock_components
        metadata_store.get.return_value = None
        from zori.agents.summarization import make_summarization_node
        node = make_summarization_node(zotero_client, llm, metadata_store, lexical_index)
        state = _fresh_state(target_key="MISSING", messages=[HumanMessage(content="summarize")])
        result = node(state)
        assert "couldn't find" in result["response"].lower()
        assert "summary" not in result

    def test_summarization_no_text_returns_error(self, mock_components):
        metadata_store, lexical_index, zotero_client, llm = mock_components
        lexical_index.get_full_text.return_value = "   "
        from zori.agents.summarization import make_summarization_node
        node = make_summarization_node(zotero_client, llm, metadata_store, lexical_index)
        state = _fresh_state(target_key="P1", messages=[HumanMessage(content="summarize")])
        result = node(state)
        assert "no text" in result["response"].lower()

    def test_summarization_llm_failure_returns_error(self, mock_components):
        metadata_store, lexical_index, zotero_client, llm = mock_components
        structured_llm = MagicMock()
        structured_llm.invoke.side_effect = RuntimeError("LLM timeout")
        llm.with_structured_output.return_value = structured_llm
        from zori.agents.summarization import make_summarization_node
        node = make_summarization_node(zotero_client, llm, metadata_store, lexical_index)
        state = _fresh_state(target_key="P1", messages=[HumanMessage(content="summarize")])
        result = node(state)
        assert "summarization failed" in result["response"].lower()
