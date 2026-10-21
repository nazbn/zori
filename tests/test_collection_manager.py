from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from zori.agents.collection_manager import (
    HISTORY_MESSAGES,
    collection_respond_node,
    make_collection_agent_node,
    make_collection_tools,
    make_confirm_collection_node,
    route_collection_agent,
    tool_error_message,
)
from zori.ingestion.zotero import ZoteroCollection, ZoteroItem
from zori.retrieval.search import SearchResult


def _result(key="P1"):
    return SearchResult(item_key=key, title=key, authors=[], year=None, journal=None, text="", score=0.9)


def _collection(name="SR", num_items=3, key="C1", parent_key=None):
    return ZoteroCollection(key=key, name=name, parent_key=parent_key, num_items=num_items)


def _item(key="P1", title="Paper"):
    return ZoteroItem(key=key, title=title, authors=["A. Author"], year="2021")


def _ai_with_calls(*calls):
    """AIMessage carrying tool calls, as the collection agent would store it."""
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": f"call_{i}"} for i, (name, args) in enumerate(calls)
    ])


def _run_tool(zot, name, args, search_results=None):
    """Run one tool call through a real ToolNode (so InjectedState / Command are exercised).

    ToolNode needs the graph runtime in LangGraph 1.x, so it runs inside a one-node graph.
    Returns (tool result text, final state).
    """
    from zori.agents.graph import ZoriState

    graph = StateGraph(ZoriState)
    graph.add_node("tools", ToolNode(make_collection_tools(zot), messages_key="collection_messages",
                                     handle_tool_errors=tool_error_message))
    graph.add_edge(START, "tools")
    state = graph.compile().invoke({
        "collection_messages": [_ai_with_calls((name, args))],
        "search_results": search_results or [],
    })
    return state["collection_messages"][-1].content, state


# ---------------------------------------------------------------------------
# create_collection tool
# ---------------------------------------------------------------------------

class TestCreateCollection:
    def test_creates_new_collection(self):
        zot = MagicMock()
        zot.find_collection.return_value = None
        zot.create_collection.return_value = "NEW1"
        text, _ = _run_tool(zot, "create_collection", {"name": "DEM"})
        zot.create_collection.assert_called_once_with("DEM", parent_key=None)
        assert "DEM" in text

    def test_reports_when_already_exists(self):
        zot = MagicMock()
        zot.find_collection.return_value = _collection("DEM")
        text, _ = _run_tool(zot, "create_collection", {"name": "DEM"})
        zot.create_collection.assert_not_called()
        assert "already exists" in text

    def test_nested_collection_uses_parent_key(self):
        zot = MagicMock()
        parent = _collection("Super-resolution")
        zot.find_collection.side_effect = lambda path: None if "/" in path else parent
        zot.create_collection.return_value = "NEW2"
        _run_tool(zot, "create_collection", {"name": "DEM", "parent_path": "Super-resolution"})
        zot.create_collection.assert_called_once_with("DEM", parent_key="C1")

    def test_nested_collection_parent_not_found(self):
        zot = MagicMock()
        zot.find_collection.return_value = None
        text, _ = _run_tool(zot, "create_collection", {"name": "DEM", "parent_path": "Missing"})
        zot.create_collection.assert_not_called()
        assert "not found" in text


# ---------------------------------------------------------------------------
# list_collections tool
# ---------------------------------------------------------------------------

class TestListCollections:
    def test_lists_all_collections(self):
        zot = MagicMock()
        zot.fetch_collections.return_value = [_collection("SR"), _collection("DEM", key="C2")]
        text, _ = _run_tool(zot, "list_collections", {})
        assert "SR" in text and "DEM" in text

    def test_lists_none_when_empty(self):
        zot = MagicMock()
        zot.fetch_collections.return_value = []
        text, _ = _run_tool(zot, "list_collections", {})
        assert "don't have any collections" in text.lower()

    def test_checks_existing_collection(self):
        zot = MagicMock()
        zot.fetch_collections.return_value = [_collection("SR", num_items=3)]
        zot.find_collection.return_value = _collection("SR", num_items=3)
        text, _ = _run_tool(zot, "list_collections", {"name": "SR"})
        assert "SR" in text and "3" in text

    def test_checks_missing_collection(self):
        zot = MagicMock()
        zot.fetch_collections.return_value = []
        zot.find_collection.return_value = None
        text, _ = _run_tool(zot, "list_collections", {"name": "Missing"})
        assert "no collection" in text.lower()

    def test_checks_subcollection_shows_full_path(self):
        zot = MagicMock()
        parent = _collection("Super-resolution", key="P1")
        child = _collection("DEM", key="C1", parent_key="P1")
        zot.fetch_collections.return_value = [parent, child]
        zot.find_collection.return_value = child
        text, _ = _run_tool(zot, "list_collections", {"name": "DEM"})
        assert "Super-resolution/DEM" in text

    def test_lists_tree_sorted_and_nested_with_counts(self):
        zot = MagicMock()
        # deliberately out of order, children before parents, mixed case
        zot.fetch_collections.return_value = [
            _collection("DEM-SR-Attention", num_items=4, key="C3", parent_key="C1"),
            _collection("Test", num_items=2, key="T1"),
            _collection("DEM", num_items=8, key="C1", parent_key="P1"),
            _collection("benchmarking", num_items=6, key="C2", parent_key="P1"),
            _collection("Super-resolution", num_items=20, key="P1"),
            _collection("AI agents", num_items=3, key="A1"),
        ]
        text, _ = _run_tool(zot, "list_collections", {})
        assert text.split("\n") == [
            "You have 6 collection(s):",
            "AI agents (3 items)",
            "Super-resolution (20 items)",
            "├─ benchmarking (6 items)",
            "└─ DEM (8 items)",
            "   └─ DEM-SR-Attention (4 items)",
            "Test (2 items)",
        ]

    def test_vertical_line_continues_under_non_last_child(self):
        zot = MagicMock()
        zot.fetch_collections.return_value = [
            _collection("Root", key="R"),
            _collection("A", key="A", parent_key="R"),
            _collection("A1", key="A1", parent_key="A"),
            _collection("B", key="B", parent_key="R"),
        ]
        text, _ = _run_tool(zot, "list_collections", {})
        assert text.split("\n")[1:] == [
            "Root (3 items)",
            "├─ A (3 items)",
            "│  └─ A1 (3 items)",
            "└─ B (3 items)",
        ]

    def test_singular_item_count(self):
        zot = MagicMock()
        zot.fetch_collections.return_value = [_collection("One", num_items=1)]
        text, _ = _run_tool(zot, "list_collections", {})
        assert "One (1 item)" in text.split("\n")

    def test_orphaned_subcollection_shown_at_top_level(self):
        zot = MagicMock()
        zot.fetch_collections.return_value = [_collection("Lost", key="C1", parent_key="GONE")]
        text, _ = _run_tool(zot, "list_collections", {})
        assert "Lost (3 items)" in text.split("\n")


# ---------------------------------------------------------------------------
# list_collection_items tool
# ---------------------------------------------------------------------------

class TestListCollectionItems:
    def test_lists_items_and_updates_search_results(self):
        zot = MagicMock()
        zot.find_collection.return_value = _collection("RAG", num_items=2)
        zot.fetch_collection_items.return_value = [_item("P1", "Paper One"), _item("P2", "Paper Two")]
        text, state = _run_tool(zot, "list_collection_items", {"collection_name": "RAG"},
                                search_results=[_result("OLD")])
        assert "Paper One" in text and "Paper Two" in text
        assert [r.item_key for r in state["search_results"]] == ["P1", "P2"]

    def test_missing_collection_keeps_search_results(self):
        zot = MagicMock()
        zot.find_collection.return_value = None
        text, state = _run_tool(zot, "list_collection_items", {"collection_name": "Missing"},
                                search_results=[_result("OLD")])
        assert "no collection" in text.lower()
        assert [r.item_key for r in state["search_results"]] == ["OLD"]

    def test_empty_collection_keeps_search_results(self):
        zot = MagicMock()
        zot.find_collection.return_value = _collection("Empty", num_items=0)
        zot.fetch_collection_items.return_value = []
        text, state = _run_tool(zot, "list_collection_items", {"collection_name": "Empty"},
                                search_results=[_result("OLD")])
        assert "no items" in text.lower()
        assert [r.item_key for r in state["search_results"]] == ["OLD"]


# ---------------------------------------------------------------------------
# add_to_collection tool (runs only after confirmation — no interrupt inside)
# ---------------------------------------------------------------------------

class TestAddToCollection:
    def test_no_results_returns_error(self):
        zot = MagicMock()
        text, _ = _run_tool(zot, "add_to_collection", {"collection_name": "SR"}, search_results=[])
        zot.add_items_to_collection.assert_not_called()
        assert "no search results" in text.lower()

    def test_new_collection_is_created_and_populated(self):
        zot = MagicMock()
        zot.find_collection.return_value = None
        zot.create_collection.return_value = "NEW1"
        text, _ = _run_tool(zot, "add_to_collection", {"collection_name": "SR"},
                            search_results=[_result("P1"), _result("P2")])
        zot.create_collection.assert_called_once_with("SR")
        zot.add_items_to_collection.assert_called_once_with("NEW1", ["P1", "P2"])
        assert "Created" in text

    def test_existing_collection_adds_papers(self):
        zot = MagicMock()
        zot.find_collection.return_value = _collection("SR", num_items=3)
        text, _ = _run_tool(zot, "add_to_collection", {"collection_name": "SR"}, search_results=[_result("P1")])
        zot.create_collection.assert_not_called()
        zot.add_items_to_collection.assert_called_once_with("C1", ["P1"])
        assert "Added 1 paper" in text

    def test_selection_adds_only_chosen_results(self):
        zot = MagicMock()
        zot.find_collection.return_value = _collection("SR", num_items=0)
        _run_tool(zot, "add_to_collection", {"collection_name": "SR", "selection": "1,3"},
                  search_results=[_result("P1"), _result("P2"), _result("P3")])
        zot.add_items_to_collection.assert_called_once_with("C1", ["P1", "P3"])

    def test_invalid_selection_returns_error(self):
        zot = MagicMock()
        text, _ = _run_tool(zot, "add_to_collection", {"collection_name": "SR", "selection": "99"},
                            search_results=[_result("P1")])
        zot.add_items_to_collection.assert_not_called()
        assert "99" in text

    def test_zotero_error_becomes_tool_result(self):
        zot = MagicMock()
        zot.find_collection.side_effect = RuntimeError("503 Service Unavailable")
        text, _ = _run_tool(zot, "add_to_collection", {"collection_name": "SR"}, search_results=[_result("P1")])
        assert "Zotero request failed" in text
        assert "503" in text


# ---------------------------------------------------------------------------
# route_collection_agent
# ---------------------------------------------------------------------------

class TestRouteCollectionAgent:
    def test_no_tool_calls_goes_to_respond(self):
        state = {"collection_messages": [AIMessage(content="Which collection do you mean?")]}
        assert route_collection_agent(state) == "collection_respond"

    def test_read_only_tool_goes_straight_to_tools(self):
        state = {"collection_messages": [_ai_with_calls(("list_collections", {}))]}
        assert route_collection_agent(state) == "collection_tools"

    def test_add_goes_to_confirmation(self):
        state = {"collection_messages": [_ai_with_calls(("add_to_collection", {"collection_name": "SR"}))]}
        assert route_collection_agent(state) == "confirm_collection"


# ---------------------------------------------------------------------------
# collection_agent node
# ---------------------------------------------------------------------------

class TestCollectionAgent:
    def _sent(self, messages, search_results=None):
        """Run the node and return the message list it sent to the LLM."""
        tool_llm = MagicMock()
        tool_llm.invoke.return_value = AIMessage(content="ok")
        llm = MagicMock()
        llm.bind_tools.return_value = tool_llm
        node = make_collection_agent_node(llm, [])
        node({"messages": messages, "query": messages[-1].content, "search_results": search_results or []})
        return tool_llm.invoke.call_args[0][0]

    def test_sends_system_prompt_then_conversation(self):
        messages = [
            HumanMessage(content="add 1 and 3 to a collection called Test"),
            AIMessage(content="Create collection 'Test' and add 2 paper(s) to it? (yes / no)"),
            HumanMessage(content="no let's add 1 and 2"),
        ]
        sent = self._sent(messages)
        assert isinstance(sent[0], SystemMessage)
        assert [m.content for m in sent[1:]] == [m.content for m in messages]

    def test_search_results_go_in_system_message(self):
        sent = self._sent([HumanMessage(content="add 2 to SR")], [_result("P1"), _result("P2")])
        assert "2. P2 (P2)" in sent[0].content
        assert "2. P2" not in sent[-1].content  # the user's message stays as they wrote it

    def test_history_is_trimmed_and_starts_on_human(self):
        messages = []
        for i in range(10):
            messages += [HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")]
        messages.append(HumanMessage(content="latest"))
        history = self._sent(messages)[1:]
        assert len(history) <= HISTORY_MESSAGES
        assert isinstance(history[0], HumanMessage)
        assert history[-1].content == "latest"


# ---------------------------------------------------------------------------
# confirm_collection node
# ---------------------------------------------------------------------------

class TestConfirmCollection:
    def _run(self, zot, reply, args=None, search_results=None):
        node = make_confirm_collection_node(zot)
        state = {
            "collection_messages": [_ai_with_calls(("add_to_collection", args or {"collection_name": "SR"}))],
            "search_results": search_results if search_results is not None else [_result("P1"), _result("P2")],
        }
        with patch("zori.agents.collection_manager.interrupt", return_value=reply) as mock_interrupt:
            cmd = node(state)
        return cmd, mock_interrupt

    def test_new_collection_question(self):
        zot = MagicMock()
        zot.find_collection.return_value = None
        _, mock_interrupt = self._run(zot, "no")
        text = mock_interrupt.call_args[0][0]["text"]
        assert "Create collection 'SR'" in text
        assert "2 paper(s)" in text

    def test_existing_collection_question_shows_counts(self):
        zot = MagicMock()
        zot.find_collection.return_value = _collection("SR", num_items=5)
        _, mock_interrupt = self._run(zot, "no")
        text = mock_interrupt.call_args[0][0]["text"]
        assert "SR" in text and "5 item(s)" in text and "2 paper(s)" in text

    def test_selection_count_in_question(self):
        zot = MagicMock()
        zot.find_collection.return_value = None
        _, mock_interrupt = self._run(zot, "no", args={"collection_name": "SR", "selection": "1"})
        assert "1 paper(s)" in mock_interrupt.call_args[0][0]["text"]

    def test_yes_goes_to_tools(self):
        zot = MagicMock()
        zot.find_collection.return_value = None
        cmd, _ = self._run(zot, "yes")
        assert cmd.goto == "collection_tools"
        zot.add_items_to_collection.assert_not_called()  # the tool does the work, not this node

    def test_no_ends_without_adding(self):
        zot = MagicMock()
        zot.find_collection.return_value = None
        cmd, _ = self._run(zot, "no")
        assert cmd.goto == END
        assert "nothing was added" in cmd.update["response"].lower()

    def test_other_reply_redirects_to_router(self):
        zot = MagicMock()
        zot.find_collection.return_value = None
        cmd, _ = self._run(zot, "actually add them to DEM instead")
        assert cmd.goto == "router"
        assert cmd.update["query"] == "actually add them to DEM instead"

    def test_zotero_error_ends_turn_without_asking(self):
        zot = MagicMock()
        zot.find_collection.side_effect = Exception("Code: 503")
        cmd, mock_interrupt = self._run(zot, "yes")
        mock_interrupt.assert_not_called()
        assert cmd.goto == END
        assert "Zotero request failed: Code: 503" in cmd.update["response"]
        assert "Nothing was added" in cmd.update["response"]

    def test_nothing_to_add_skips_question(self):
        zot = MagicMock()
        cmd, mock_interrupt = self._run(zot, "yes", search_results=[])
        mock_interrupt.assert_not_called()
        assert cmd.goto == "collection_tools"  # the tool reports "no search results"


# ---------------------------------------------------------------------------
# collection_respond node
# ---------------------------------------------------------------------------

class TestCollectionRespond:
    def test_joins_tool_results(self):
        state = {"collection_messages": [
            _ai_with_calls(("list_collections", {})),
            ToolMessage(content="You have 2 collection(s): SR, DEM.", tool_call_id="call_0"),
        ]}
        result = collection_respond_node(state)
        assert result["response"] == "You have 2 collection(s): SR, DEM."
        assert isinstance(result["messages"][0], AIMessage)

    def test_uses_llm_text_when_no_tools_called(self):
        state = {"collection_messages": [AIMessage(content="Which collection do you mean?")]}
        assert collection_respond_node(state)["response"] == "Which collection do you mean?"


# ---------------------------------------------------------------------------
# Full graph: interrupt + resume with a real checkpointer
# ---------------------------------------------------------------------------

class TestCollectionGraph:
    """Runs the compiled graph with MemorySaver. Guards against resume re-running the LLM."""

    def _build(self, tool_calls):
        from zori.agents.graph import build_graph
        from zori.agents.router import RouterOutput

        router_llm = MagicMock()
        router_llm.invoke.return_value = RouterOutput(intent="collection")
        tool_llm = MagicMock()
        tool_llm.invoke.return_value = _ai_with_calls(*tool_calls)
        llm = MagicMock()
        llm.with_structured_output.return_value = router_llm
        llm.bind_tools.return_value = tool_llm

        zot = MagicMock()
        zot.find_collection.return_value = None
        zot.create_collection.return_value = "NEW1"
        graph = build_graph(MagicMock(), zot, llm, MagicMock(), MagicMock())
        return graph, tool_llm, zot

    def _start(self, graph, text, search_results=None):
        config = {"configurable": {"thread_id": text}}
        state = {"messages": [HumanMessage(content=text)], "query": text}
        if search_results is not None:
            state["search_results"] = search_results
        return graph.invoke(state, config), config

    def test_add_yes_runs_approved_call_once(self):
        graph, tool_llm, zot = self._build([("add_to_collection", {"collection_name": "SR", "selection": "2"})])
        result, config = self._start(graph, "add the second one to SR", [_result("P1"), _result("P2")])
        assert "Create collection 'SR'" in result["__interrupt__"][0].value["text"]
        zot.add_items_to_collection.assert_not_called()

        result = graph.invoke(Command(resume="yes"), config)
        assert tool_llm.invoke.call_count == 1
        zot.add_items_to_collection.assert_called_once_with("NEW1", ["P2"])
        assert result["response"] == "Created collection 'SR' and added 1 paper(s)."
        # tool chatter stays out of the main conversation history
        assert not any(isinstance(m, ToolMessage) for m in result["messages"])

    def test_add_before_any_search_explains_instead_of_crashing(self):
        graph, _, zot = self._build([("add_to_collection", {"collection_name": "SR"})])
        result, _ = self._start(graph, "add these to SR")  # fresh session: no search_results key yet
        assert "__interrupt__" not in result
        assert "no search results" in result["response"].lower()
        zot.add_items_to_collection.assert_not_called()

    def test_add_no_adds_nothing(self):
        graph, tool_llm, zot = self._build([("add_to_collection", {"collection_name": "SR"})])
        _, config = self._start(graph, "add these to SR", [_result("P1")])
        result = graph.invoke(Command(resume="no"), config)
        zot.create_collection.assert_not_called()
        zot.add_items_to_collection.assert_not_called()
        assert "nothing was added" in result["response"].lower()

    def test_read_only_tool_runs_without_asking(self):
        graph, _, zot = self._build([("list_collections", {})])
        zot.fetch_collections.return_value = [_collection("SR")]
        result, _ = self._start(graph, "list my collections")
        assert "__interrupt__" not in result
        assert "SR" in result["response"]

    def test_list_items_then_results_available_for_next_turn(self):
        graph, _, zot = self._build([("list_collection_items", {"collection_name": "RAG"})])
        zot.find_collection.return_value = _collection("RAG", num_items=1)
        zot.fetch_collection_items.return_value = [_item("P9", "Paper Nine")]
        result, _ = self._start(graph, "what's in RAG")
        assert "Paper Nine" in result["response"]
        assert [r.item_key for r in result["search_results"]] == ["P9"]

    def test_zotero_outage_in_confirm_ends_turn_cleanly(self):
        graph, _, zot = self._build([("add_to_collection", {"collection_name": "SR"})])
        zot.find_collection.side_effect = Exception("Code: 503")
        result, config = self._start(graph, "add these to SR", [_result("P1")])
        assert "__interrupt__" not in result
        assert "Zotero request failed" in result["response"]
        assert not graph.get_state(config).next  # nothing left pending for the next message

    def test_crashed_turn_leaves_no_pending_question(self):
        """A node that raises leaves snapshot.next set but no interrupt — callers must check
        .interrupts (as cli.py / server.py do), and the next message must start a fresh turn."""
        graph, tool_llm, zot = self._build([("add_to_collection", {"collection_name": "SR"})])
        zot.find_collection.side_effect = None
        with patch("zori.agents.collection_manager.resolve_selection", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                self._start(graph, "add these to SR", [_result("P1")])
        config = {"configurable": {"thread_id": "add these to SR"}}
        snapshot = graph.get_state(config)
        assert snapshot.next == ("confirm_collection",)
        assert not snapshot.interrupts

        result = graph.invoke({"messages": [HumanMessage(content="yes")], "query": "yes"}, config)
        assert tool_llm.invoke.call_count == 2  # new turn went through the agent again
        assert "Create collection 'SR'" in result["__interrupt__"][0].value["text"]
        zot.add_items_to_collection.assert_not_called()

    def test_redirect_keeps_collection_name_from_earlier_turn(self):
        """The bug from manual testing: after "no let's add 1 and 2" the agent only saw that
        reply, didn't know the collection was called Test, and made up a name."""
        graph, tool_llm, _ = self._build([("add_to_collection", {"collection_name": "Test", "selection": "1,3"})])
        _, config = self._start(graph, "add 1 and 3 to a collection called Test",
                                [_result("P1"), _result("P2"), _result("P3")])
        graph.invoke(Command(resume="no let's add 1 and 2"), config)

        assert tool_llm.invoke.call_count == 2
        second_call = tool_llm.invoke.call_args_list[1][0][0]
        contents = [m.content for m in second_call]
        assert "add 1 and 3 to a collection called Test" in contents
        assert contents[-1] == "no let's add 1 and 2"

    def test_second_turn_clears_previous_tool_messages(self):
        graph, tool_llm, zot = self._build([("list_collections", {})])
        zot.fetch_collections.return_value = [_collection("SR")]
        _, config = self._start(graph, "list my collections")
        result = graph.invoke({"messages": [HumanMessage(content="again")], "query": "again"}, config)
        tool_messages = [m for m in result["collection_messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
