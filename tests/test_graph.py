from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from zori.agents.graph import ZoriState
from zori.agents.summarization import SummaryOutput


def _fresh_state(**overrides) -> dict:
    base = {
        "messages": [],
        "query": "",
        "intent": "",
        "target_key": None,
        "search_results": [],
        "summary": None,
        "response": None,
        "pending_confirmation": False,
        "confirmation_type": None,
        "candidate_key": None,
    }
    base.update(overrides)
    return base


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

    def test_find_for_summarize_asks_confirmation(self, search_service, mock_llm):
        from zori.agents.paper_finder import make_paper_finder_node
        node = make_paper_finder_node(search_service, mock_llm)
        state = _fresh_state(query="attention paper", intent="summarize",
                             messages=[HumanMessage(content="attention paper")])
        result = node(state)
        assert result["pending_confirmation"] is True
        assert result["confirmation_type"] == "paper_selection"
        assert result["candidate_key"] == "P1"

    def test_confirmation_yes_sets_target_key(self, search_service, mock_llm):
        from zori.agents.paper_finder import make_paper_finder_node
        from zori.retrieval.search import SearchResult
        node = make_paper_finder_node(search_service, mock_llm)
        result_item = SearchResult(
            item_key="P1", title="Test", authors=[], year=None,
            journal=None, text="", score=0.9,
        )
        state = _fresh_state(
            query="yes",
            pending_confirmation=True,
            confirmation_type="paper_selection",
            candidate_key="P1",
            search_results=[result_item],
            messages=[HumanMessage(content="yes")],
        )
        result = node(state)
        assert result["target_key"] == "P1"
        assert result["pending_confirmation"] is False

    def test_confirmation_no_clears_state(self, search_service, mock_llm):
        from zori.agents.paper_finder import make_paper_finder_node
        from zori.retrieval.search import SearchResult
        node = make_paper_finder_node(search_service, mock_llm)
        state = _fresh_state(
            query="no",
            pending_confirmation=True,
            confirmation_type="paper_selection",
            candidate_key="P1",
            search_results=[],
            messages=[HumanMessage(content="no")],
        )
        result = node(state)
        assert result["target_key"] is None
        assert result["pending_confirmation"] is False


# --- Writer ---

class TestWriter:
    @pytest.fixture
    def zotero_client(self):
        return MagicMock()

    def test_yes_saves_note(self, zotero_client):
        from zori.agents.writer import make_writer_node
        node = make_writer_node(zotero_client)
        state = _fresh_state(
            query="yes",
            target_key="P1",
            summary={"overview": "Great paper.", "contributions": ["A", "B"],
                     "methods": "ML", "findings": "Good results."},
            pending_confirmation=True,
            confirmation_type="save_summary",
            messages=[HumanMessage(content="yes")],
        )
        result = node(state)
        zotero_client.write_note.assert_called_once()
        assert result["pending_confirmation"] is False
        assert "saved" in result["response"].lower()

    def test_no_skips_save(self, zotero_client):
        from zori.agents.writer import make_writer_node
        node = make_writer_node(zotero_client)
        state = _fresh_state(
            query="no",
            target_key="P1",
            summary={"overview": "x", "contributions": [], "methods": "x", "findings": "x"},
            pending_confirmation=True,
            confirmation_type="save_summary",
            messages=[HumanMessage(content="no")],
        )
        result = node(state)
        zotero_client.write_note.assert_not_called()
        assert result["pending_confirmation"] is False

    def test_write_failure_returns_error_response(self, zotero_client):
        from zori.agents.writer import make_writer_node
        zotero_client.write_note.side_effect = RuntimeError("API error")
        node = make_writer_node(zotero_client)
        state = _fresh_state(
            query="yes",
            target_key="P1",
            summary={"overview": "x", "contributions": [], "methods": "x", "findings": "x"},
            pending_confirmation=True,
            confirmation_type="save_summary",
            messages=[HumanMessage(content="yes")],
        )
        result = node(state)
        assert result["pending_confirmation"] is False
        assert "failed" in result["response"].lower()


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

    def test_summarization_produces_structured_output(self, mock_components):
        metadata_store, lexical_index, zotero_client, llm = mock_components
        from zori.agents.summarization import make_summarization_node
        node = make_summarization_node(zotero_client, llm, metadata_store, lexical_index)
        state = _fresh_state(
            target_key="P1",
            messages=[HumanMessage(content="summarize this")],
        )
        result = node(state)
        assert result["summary"] is not None
        assert result["pending_confirmation"] is True
        assert result["confirmation_type"] == "save_summary"
        assert "overview" in result["summary"]

    def test_summarization_response_includes_title_and_sections(self, mock_components):
        metadata_store, lexical_index, zotero_client, llm = mock_components
        from zori.agents.summarization import make_summarization_node
        node = make_summarization_node(zotero_client, llm, metadata_store, lexical_index)
        state = _fresh_state(target_key="P1", messages=[HumanMessage(content="summarize")])
        result = node(state)
        assert "Test Paper" in result["response"]
        assert "Overview" in result["response"]
        assert "Findings" in result["response"]

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
