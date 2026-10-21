"""Collection manager: an LLM with Zotero collection tools, run as four graph nodes.

    collection_agent    LLM decides which tools to call (tool calls stored in collection_messages)
    confirm_collection  asks before add_to_collection runs (interrupt lives here, not in the tool)
    collection_tools    prebuilt ToolNode — runs the stored tool calls
    collection_respond  turns the tool results into the user-facing response

Tool messages live in their own state key, collection_messages, so they never reach
state["messages"]: the router sends that history to an LLM without tools bound, and
Anthropic rejects tool results without tool definitions.
"""
import structlog
from typing import Annotated, Callable, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.graph import END
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt

from zori.agents.confirm import is_no, is_yes, redirect_to_router
from zori.agents.graph import ZoriState
from zori.display.rich import format_authors
from zori.ingestion.zotero import ZoteroClient, ZoteroCollection
from zori.retrieval.search import SearchResult

logger = structlog.get_logger()

SYSTEM_PROMPT = """You are Zori, a research assistant for a personal Zotero library.
The user wants to manage or query their Zotero collections.

You have four tools:
- add_to_collection: Add papers from the current search results to a collection, creating it if it
  doesn't exist. Pass selection="all" (default) to add every result, or a comma-separated list of
  result numbers (e.g. "1" or "1,3") to add only the ones the user referred to. The user is asked to
  confirm before anything is added — you don't need to ask for confirmation yourself.
- create_collection: Create a new empty collection (optionally nested under a parent path like
  "Parent/Child"). Use this ONLY when the user explicitly wants an empty collection, with no papers.
- list_collections: List all collections (shown as "Parent/Child" paths), or check whether a specific
  collection exists by name or path. Pass name=None to list everything; pass a name to look up one.
- list_collection_items: List the papers inside a specific collection, by name or "Parent/Child" path.

Rules:
- If the user wants to add papers to a collection (whether new or existing), use add_to_collection only.
  Do NOT call create_collection first — add_to_collection creates the collection if needed.
- If the user refers to specific results by number (e.g. "add 1 and 3", "the first one"), pass those
  numbers as selection; otherwise use selection="all".
- Do not call add_to_collection if there are no search results available.
- Use list_collections for questions about which collections exist; use list_collection_items when the
  user wants to see what papers are inside a specific collection."""

# Tools that change the user's library and need a yes from the user before they run
NEEDS_CONFIRMATION = {"add_to_collection"}


def _full_path(col: ZoteroCollection, by_key: dict[str, ZoteroCollection]) -> str:
    """Build a 'Parent/Child' path string for a collection by walking up parent_key."""
    parts = [col.name]
    parent_key = col.parent_key
    while parent_key and parent_key in by_key:
        parent = by_key[parent_key]
        parts.append(parent.name)
        parent_key = parent.parent_key
    return "/".join(reversed(parts))


def resolve_selection(search_results: list[SearchResult], selection: str | None) -> list[str]:
    """Map 'all' or '1,3' to item keys of the current search results; invalid numbers are skipped."""
    item_keys = [r.item_key for r in search_results]
    if not selection or selection.strip().lower() == "all":
        return item_keys
    keys = []
    for part in selection.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(item_keys):
                keys.append(item_keys[idx])
    return keys


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def make_collection_tools(zotero_client: ZoteroClient) -> list[BaseTool]:
    """Build the collection tools, bound to a Zotero client.

    InjectedState / InjectedToolCallId parameters are filled in by ToolNode and hidden from the LLM.
    """

    @tool
    def create_collection(name: str, parent_path: str | None = None) -> str:
        """Create a new Zotero collection. Use parent_path (e.g. 'Super-resolution/DEM') to nest it."""
        lookup = f"{parent_path}/{name}" if parent_path else name
        if zotero_client.find_collection(lookup) is not None:
            return f"Collection '{name}' already exists."
        parent_key = None
        if parent_path:
            parent = zotero_client.find_collection(parent_path)
            if parent is None:
                return f"Parent collection '{parent_path}' not found."
            parent_key = parent.key
        col_key = zotero_client.create_collection(name, parent_key=parent_key)
        logger.debug("collection_created", name=name, key=col_key)
        return f"Created collection '{name}'."

    @tool
    def add_to_collection(
        collection_name: str,
        state: Annotated[dict, InjectedState],
        selection: str = "all",
    ) -> str:
        """Add papers from the current search results to a Zotero collection, creating it if needed.
        selection: 'all' (default) or comma-separated result numbers, e.g. '1' or '1,3'."""
        # Whole state rather than InjectedState("search_results"): that form raises KeyError
        # before the first search of a session, when the key doesn't exist yet.
        search_results = state.get("search_results", [])
        if not search_results:
            return "No search results to add. Search for papers first, then ask me to add them to a collection."
        keys = resolve_selection(search_results, selection)
        if not keys:
            return f"Couldn't match selection '{selection}' to any of the {len(search_results)} search result(s)."

        col = zotero_client.find_collection(collection_name)
        if col is None:
            col_key = zotero_client.create_collection(collection_name)
            zotero_client.add_items_to_collection(col_key, keys)
            logger.debug("collection_created_and_populated", name=collection_name, count=len(keys))
            return f"Created collection '{collection_name}' and added {len(keys)} paper(s)."
        zotero_client.add_items_to_collection(col.key, keys)
        logger.debug("items_added_to_collection", name=collection_name, count=len(keys))
        return f"Added {len(keys)} paper(s) to '{collection_name}'."

    @tool
    def list_collections(name: str | None = None) -> str:
        """List all Zotero collections as 'Parent/Child' paths, or check whether one exists by name/path."""
        cols = zotero_client.fetch_collections()
        by_key = {c.key: c for c in cols}
        if name:
            col = zotero_client.find_collection(name)
            if col is None:
                return f"No collection named '{name}' found."
            return f"'{_full_path(col, by_key)}' exists ({col.num_items} item(s))."
        if not cols:
            return "You don't have any collections yet."
        paths = ", ".join(_full_path(c, by_key) for c in cols)
        return f"You have {len(cols)} collection(s): {paths}."

    @tool
    def list_collection_items(
        collection_name: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str | Command:
        """List the papers inside a Zotero collection, by name or 'Parent/Child' path."""
        col = zotero_client.find_collection(collection_name)
        if col is None:
            return f"No collection named '{collection_name}' found."
        items = zotero_client.fetch_collection_items(col.key)
        if not items:
            return f"'{col.name}' has no items."
        lines = [f"'{col.name}' ({len(items)} item(s)):"]
        for i, it in enumerate(items, 1):
            lines.append(f"  {i}. {it.title} — {format_authors(it.authors)} ({it.year or '?'})")
        # Command lets a tool update state beyond its own result: the listed papers become the
        # current search results, so "summarize 2" or "add 1 to X" works next.
        return Command(update={
            "search_results": [
                SearchResult(
                    item_key=it.key, title=it.title, authors=it.authors,
                    year=it.year, journal=it.journal, text="", score=1.0,
                )
                for it in items
            ],
            "collection_messages": [ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)],
        })

    return [create_collection, add_to_collection, list_collections, list_collection_items]


def tool_error_message(e: Exception) -> str:
    """ToolNode error handler: a failing Zotero call becomes the tool's result instead of crashing the graph."""
    logger.warning("collection_tool_failed", error=str(e))
    return f"Zotero request failed: {e}"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def make_collection_agent_node(llm: BaseChatModel, tools: list[BaseTool]) -> Callable[[ZoriState], dict]:
    """Return the node where the LLM picks tools. It only decides — ToolNode executes."""
    llm_with_tools = llm.bind_tools(tools)

    def collection_agent_node(state: ZoriState) -> dict:
        context = state["query"]
        search_results = state.get("search_results", [])
        if search_results:
            lines = "\n".join(f"{i}. {r.title} ({r.item_key})" for i, r in enumerate(search_results, 1))
            context += f"\n\n{len(search_results)} search result(s) currently available:\n{lines}"

        response = llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=context)])
        logger.debug("collection_agent", tool_calls=[(tc["name"], tc["args"]) for tc in response.tool_calls])
        # Clear last turn's tool messages, then store this turn's decision
        return {"collection_messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), response]}

    return collection_agent_node


def route_collection_agent(
    state: ZoriState,
) -> Literal["confirm_collection", "collection_tools", "collection_respond"]:
    """Like the prebuilt tools_condition, plus a stop at confirm_collection for library-changing tools."""
    tool_calls = getattr(state["collection_messages"][-1], "tool_calls", None)
    if not tool_calls:
        return "collection_respond"
    if any(tc["name"] in NEEDS_CONFIRMATION for tc in tool_calls):
        return "confirm_collection"
    return "collection_tools"


def make_confirm_collection_node(
    zotero_client: ZoteroClient,
) -> Callable[[ZoriState], Command[Literal["collection_tools", "router", "__end__"]]]:
    """Return the node that asks before add_to_collection runs.

    The tool calls are already stored in state, so resuming runs exactly the approved calls.
    Only read-only Zotero lookups happen before the interrupt (they repeat on resume, harmlessly).
    """

    def confirm_collection_node(state: ZoriState) -> Command[Literal["collection_tools", "router", "__end__"]]:
        search_results = state.get("search_results", [])
        tool_calls = state["collection_messages"][-1].tool_calls

        questions = []
        for tc in tool_calls:
            if tc["name"] not in NEEDS_CONFIRMATION:
                continue
            keys = resolve_selection(search_results, tc["args"].get("selection", "all"))
            if not keys:
                continue  # nothing to add — the tool will explain why
            name = tc["args"]["collection_name"]
            # This lookup runs outside ToolNode, so handle_tool_errors doesn't cover it
            try:
                col = zotero_client.find_collection(name)
            except Exception as e:
                response = tool_error_message(e) + "\nNothing was added."
                return Command(goto=END, update={"response": response, "messages": [AIMessage(content=response)]})
            if col is None:
                questions.append(f"Create collection '{name}' and add {len(keys)} paper(s) to it?")
            else:
                questions.append(f"'{name}' already exists ({col.num_items} item(s)). Add {len(keys)} paper(s) to it?")

        if not questions:
            return Command(goto="collection_tools")

        question = "\n".join(questions) + " (yes / no)"
        reply = interrupt({"text": question, "zotero_key": None})

        if is_yes(reply):
            return Command(goto="collection_tools")
        if is_no(reply):
            response = "No problem — nothing was added."
            return Command(goto=END, update={"response": response, "messages": [AIMessage(content=response)]})
        return redirect_to_router(question, reply)

    return confirm_collection_node


def collection_respond_node(state: ZoriState) -> dict:
    """Turn this turn's tool results (or the LLM's plain answer) into the user-facing response."""
    collection_messages = state.get("collection_messages", [])
    tool_results = [m.content for m in collection_messages if isinstance(m, ToolMessage)]
    if tool_results:
        response = "\n".join(tool_results)
    else:
        last = collection_messages[-1] if collection_messages else None
        response = (last.content if last else "") or "I'm not sure what you'd like to do with your collections."
    return {"response": response, "messages": [AIMessage(content=response)]}
