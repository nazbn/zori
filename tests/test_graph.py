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
        "candidates": [],
        "summary": None,
        "summary_text": None,
        "response": None,
    }
    base.update(overrides)
    return base


# --- Routing functions ---

from langgraph.graph import END
from zori.agents.graph import (
    _route_from_paper_finder,
    _route_from_router,
    _route_from_summarization,
)


class TestRouting:
    def _state(self, **kw):
        return {"intent": "", "target_key": None, "search_results": [], **kw}

    def test_search_goes_to_paper_finder(self):
        assert _route_from_router(self._state(intent="search")) == "paper_finder"

    def test_summarize_without_key_goes_to_paper_finder(self):
        assert _route_from_router(self._state(intent="summarize")) == "paper_finder"

    def test_summarize_with_key_goes_to_summarization(self):
        assert _route_from_router(self._state(intent="summarize", target_key="K1")) == "summarization"

    def test_collection_goes_to_collection_agent(self):
        assert _route_from_router(self._state(intent="collection")) == "collection_agent"

    def test_general_goes_to_end(self):
        assert _route_from_router(self._state(intent="general")) == END

    def test_paper_finder_with_candidates_goes_to_confirm_paper(self):
        state = self._state(intent="summarize", candidates=["P1"])
        assert _route_from_paper_finder(state) == "confirm_paper"

    def test_paper_finder_without_candidates_goes_to_end(self):
        assert _route_from_paper_finder(self._state(intent="summarize", candidates=[])) == END

    def test_paper_finder_search_intent_goes_to_end(self):
        assert _route_from_paper_finder(self._state(intent="search", candidates=["P1"])) == END

    def test_summarization_with_summary_goes_to_confirm_save(self):
        assert _route_from_summarization(self._state(summary={"overview": "x"})) == "confirm_save"

    def test_summarization_without_summary_goes_to_end(self):
        assert _route_from_summarization(self._state(summary=None)) == END


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

    def test_find_for_summarize_returns_candidates(self, search_service, mock_llm):
        from zori.agents.paper_finder import make_paper_finder_node
        node = make_paper_finder_node(search_service, mock_llm)
        state = _fresh_state(query="attention paper", intent="summarize",
                             messages=[HumanMessage(content="attention paper")])
        with patch("zori.agents.paper_finder.interrupt") as mock_interrupt:
            result = node(state)
        mock_interrupt.assert_not_called()
        assert result["candidates"][0].item_key == "P1"


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

    def test_summarization_stores_summary_without_interrupting(self, mock_components):
        metadata_store, lexical_index, zotero_client, llm = mock_components
        from zori.agents.summarization import make_summarization_node
        node = make_summarization_node(zotero_client, llm, metadata_store, lexical_index)
        state = _fresh_state(target_key="P1", messages=[HumanMessage(content="summarize this")])
        with patch("zori.agents.summarization.interrupt") as mock_interrupt:
            result = node(state)
        mock_interrupt.assert_not_called()
        zotero_client.write_note.assert_not_called()
        assert result["summary"]["overview"] == "A great paper."
        assert "Test Paper" in result["summary_text"]
        assert "Overview" in result["summary_text"]

    def test_summarization_missing_metadata_returns_error(self, mock_components):
        metadata_store, lexical_index, zotero_client, llm = mock_components
        metadata_store.get.return_value = None
        from zori.agents.summarization import make_summarization_node
        node = make_summarization_node(zotero_client, llm, metadata_store, lexical_index)
        state = _fresh_state(target_key="MISSING", messages=[HumanMessage(content="summarize")])
        result = node(state)
        assert "couldn't find" in result["response"].lower()
        assert result["summary"] is None

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
        assert result["summary"] is None


# --- Confirm save (spec for confirm_save_node) ---

class TestConfirmSave:
    SUMMARY = {
        "overview": "A great paper.",
        "contributions": ["Novel approach"],
        "methods": "Deep learning",
        "findings": "State-of-the-art results.",
    }

    def _node_and_state(self):
        from zori.agents.summarization import make_confirm_save_node
        zotero_client = MagicMock()
        state = _fresh_state(
            target_key="P1",
            summary=self.SUMMARY,
            summary_text="Test Paper — Smith (2023)\n\nOverview\nA great paper.",
        )
        return make_confirm_save_node(zotero_client), zotero_client, state

    def test_interrupt_shows_summary_text_and_zotero_key(self):
        node, _, state = self._node_and_state()
        with patch("zori.agents.summarization.interrupt", return_value="no") as mock_interrupt:
            node(state)
        value = mock_interrupt.call_args[0][0]
        assert value == {"text": state["summary_text"], "zotero_key": "P1"}

    def test_yes_saves_stored_summary_as_note(self):
        node, zotero_client, state = self._node_and_state()
        with patch("zori.agents.summarization.interrupt", return_value="yes"):
            cmd = node(state)
        zotero_client.write_note.assert_called_once()
        item_key, html = zotero_client.write_note.call_args[0][:2]
        assert item_key == "P1"
        assert "A great paper." in html
        assert zotero_client.write_note.call_args.kwargs["tags"] == ["zori-summary"]
        assert cmd.goto == END
        assert "saved" in cmd.update["response"].lower()
        assert isinstance(cmd.update["messages"][0], AIMessage)

    def test_y_also_counts_as_yes(self):
        node, zotero_client, state = self._node_and_state()
        with patch("zori.agents.summarization.interrupt", return_value=" Y "):
            node(state)
        zotero_client.write_note.assert_called_once()

    def test_no_skips_save(self):
        node, zotero_client, state = self._node_and_state()
        with patch("zori.agents.summarization.interrupt", return_value="no"):
            cmd = node(state)
        zotero_client.write_note.assert_not_called()
        assert cmd.goto == END
        assert cmd.update["response"]

    def test_write_failure_returns_error(self):
        node, zotero_client, state = self._node_and_state()
        zotero_client.write_note.side_effect = RuntimeError("API error")
        with patch("zori.agents.summarization.interrupt", return_value="yes"):
            cmd = node(state)
        assert "failed" in cmd.update["response"].lower()

    def test_other_reply_redirects_to_router(self):
        node, zotero_client, state = self._node_and_state()
        reply = "no, summarize the BERT paper instead"
        with patch("zori.agents.summarization.interrupt", return_value=reply):
            cmd = node(state)
        zotero_client.write_note.assert_not_called()
        assert cmd.goto == "router"
        assert cmd.update["query"] == reply
        question_msg, reply_msg = cmd.update["messages"]
        assert question_msg.content == state["summary_text"]
        assert reply_msg.content == reply


# --- Full graph: interrupt + resume with a real checkpointer ---

class TestSummarizeResume:
    """Runs the compiled graph with MemorySaver — no patched interrupt().

    Guards against the bug where resuming re-ran the summarization LLM call.
    """

    def _build(self):
        from zori.agents.graph import build_graph
        from zori.agents.router import RouterOutput

        router_llm = MagicMock()
        router_llm.invoke.return_value = RouterOutput(intent="summarize", target_key="P1")
        summary_llm = MagicMock()
        summary_llm.invoke.return_value = SummaryOutput(
            overview="A great paper.",
            contributions=["Novel approach"],
            methods="Deep learning",
            findings="State-of-the-art results.",
        )
        llm = MagicMock()
        llm.with_structured_output.side_effect = (
            lambda schema: router_llm if schema is RouterOutput else summary_llm
        )

        metadata_store = MagicMock()
        metadata_store.get.return_value = {"title": "Test Paper", "authors": ["Smith"], "year": "2023"}
        lexical_index = MagicMock()
        lexical_index.get_full_text.return_value = "Full paper text."
        zotero_client = MagicMock()

        graph = build_graph(MagicMock(), zotero_client, llm, lexical_index, metadata_store)
        return graph, summary_llm, zotero_client

    def _start(self, graph):
        config = {"configurable": {"thread_id": "test-thread"}}
        result = graph.invoke(
            {"messages": [HumanMessage(content="summarize P1")], "query": "summarize P1"}, config
        )
        return result, config

    def test_pauses_with_summary(self):
        graph, _, zotero_client = self._build()
        result, _ = self._start(graph)
        assert "__interrupt__" in result
        assert "Test Paper" in result["__interrupt__"][0].value["text"]
        zotero_client.write_note.assert_not_called()

    def test_resume_yes_saves_without_rerunning_llm(self):
        from langgraph.types import Command
        graph, summary_llm, zotero_client = self._build()
        _, config = self._start(graph)
        result = graph.invoke(Command(resume="yes"), config)

        assert summary_llm.invoke.call_count == 1
        zotero_client.write_note.assert_called_once()
        assert "A great paper." in zotero_client.write_note.call_args[0][1]
        assert "saved" in result["response"].lower()

    def test_resume_no_does_not_save(self):
        from langgraph.types import Command
        graph, summary_llm, zotero_client = self._build()
        _, config = self._start(graph)
        graph.invoke(Command(resume="no"), config)

        assert summary_llm.invoke.call_count == 1
        zotero_client.write_note.assert_not_called()


class TestPickPaperResume:
    """"summarize the X paper" → pick from candidates → summarize → save, on the compiled graph.

    Guards against the bug where resuming re-ran the search plan and search, so "2" could
    point at a different paper than the one shown.
    """

    def _build(self):
        from zori.agents.graph import build_graph
        from zori.agents.paper_finder import SearchPlan
        from zori.agents.router import RouterOutput
        from zori.retrieval.search import SearchResult

        router_llm = MagicMock()
        router_llm.invoke.return_value = RouterOutput(intent="summarize", target_key=None)
        planner_llm = MagicMock()
        planner_llm.invoke.return_value = SearchPlan(display_query="attention", semantic_query="attention")
        summary_llm = MagicMock()
        summary_llm.invoke.return_value = SummaryOutput(
            overview="Second paper.", contributions=[], methods="", findings="",
        )
        by_schema = {RouterOutput: router_llm, SearchPlan: planner_llm, SummaryOutput: summary_llm}
        llm = MagicMock()
        llm.with_structured_output.side_effect = lambda schema: by_schema[schema]

        search_service = MagicMock()
        search_service.hybrid_search.return_value = [
            SearchResult(item_key=k, title=f"Paper {k}", authors=[], year="2020",
                         journal=None, text="", score=1.0 - i / 10)
            for i, k in enumerate(["P1", "P2", "P3"])
        ]
        metadata_store = MagicMock()
        metadata_store.get.side_effect = lambda key: {"title": f"Paper {key}", "authors": [], "year": "2020"}
        lexical_index = MagicMock()
        lexical_index.get_full_text.return_value = "Full paper text."
        zotero_client = MagicMock()

        graph = build_graph(search_service, zotero_client, llm, lexical_index, metadata_store)
        mocks = {"planner": planner_llm, "search": search_service, "summary": summary_llm,
                 "metadata": metadata_store, "zotero": zotero_client}
        return graph, mocks

    def test_pick_then_save(self):
        from langgraph.types import Command
        graph, m = self._build()
        config = {"configurable": {"thread_id": "pick-thread"}}

        result = graph.invoke(
            {"messages": [HumanMessage(content="summarize the attention paper")],
             "query": "summarize the attention paper"},
            config,
        )
        assert "Which one?" in result["__interrupt__"][0].value["text"]

        result = graph.invoke(Command(resume="2"), config)
        # paused again, now on the save confirmation for the paper that was picked
        assert "Paper P2" in result["__interrupt__"][0].value["text"]
        assert result["target_key"] == "P2"

        result = graph.invoke(Command(resume="yes"), config)
        assert m["zotero"].write_note.call_args[0][0] == "P2"

        # every expensive step ran exactly once across two resumes
        assert m["planner"].invoke.call_count == 1
        assert m["search"].hybrid_search.call_count == 1
        assert m["summary"].invoke.call_count == 1

    def test_pick_none_ends_without_summarizing(self):
        from langgraph.types import Command
        graph, m = self._build()
        config = {"configurable": {"thread_id": "pick-none-thread"}}
        graph.invoke(
            {"messages": [HumanMessage(content="summarize the attention paper")],
             "query": "summarize the attention paper"},
            config,
        )
        result = graph.invoke(Command(resume="none"), config)

        assert "__interrupt__" not in result
        assert result["response"].startswith("No problem")
        m["summary"].invoke.assert_not_called()

    def test_other_reply_runs_as_new_request(self):
        """"not these, the paper named BERT" goes back through router → paper_finder."""
        from langgraph.types import Command
        graph, m = self._build()
        config = {"configurable": {"thread_id": "redirect-thread"}}
        graph.invoke(
            {"messages": [HumanMessage(content="summarize the attention paper")],
             "query": "summarize the attention paper"},
            config,
        )
        reply = "not these, the paper named BERT"
        result = graph.invoke(Command(resume=reply), config)

        # a fresh search ran for the new request and is asking again
        assert m["planner"].invoke.call_count == 2
        assert reply in m["planner"].invoke.call_args[0][0]
        assert "Which one?" in result["__interrupt__"][0].value["text"]
        # the router saw the question and the reply in the history
        contents = [msg.content for msg in result["messages"]]
        assert any("Which one?" in c for c in contents)
        assert contents[-1] == reply
        m["summary"].invoke.assert_not_called()
