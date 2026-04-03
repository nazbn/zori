import operator
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

from zori.ingestion.zotero import ZoteroClient
from zori.retrieval.search import SearchResult, SearchService


class ZoriState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    query: str
    intent: str                        # "search" | "summarize" | "general"
    search_mode: str                   # "display" | "find_for_summarize"
    target_key: str | None             # Zotero item key, set before summarization
    search_results: list[SearchResult]
    summary: str | None
    response: str | None
    # Human-in-the-loop confirmation — one generic mechanism for all confirmation types
    pending_confirmation: bool
    confirmation_type: str | None      # "paper_selection" | "save_summary" | ...
    candidate_key: str | None          # used by paper_selection confirmation


def _route_from_router(state: ZoriState) -> str:
    if state.get("pending_confirmation"):
        ctype = state.get("confirmation_type")
        if ctype == "paper_selection":
            return "paper_finder"
        if ctype == "save_summary":
            return "writer"
    intent = state["intent"]
    if intent == "summarize" and not state.get("target_key"):
        return "paper_finder"
    return intent


def _route_from_paper_finder(state: ZoriState) -> str:
    if state.get("target_key") and state.get("intent") == "summarize":
        return "summarization"
    return END


def build_graph(
    search_service: SearchService,
    zotero_client: ZoteroClient,
    llm: BaseChatModel,
    max_search_iterations: int = 3,
):
    from zori.agents.paper_finder import make_paper_finder_node
    from zori.agents.router import make_router_node
    from zori.agents.summarization import make_summarization_node
    from zori.agents.writer import make_writer_node

    graph = StateGraph(ZoriState)

    graph.add_node("router", make_router_node(llm))
    graph.add_node("paper_finder", make_paper_finder_node(search_service, llm, max_search_iterations))
    graph.add_node("summarization", make_summarization_node(zotero_client, llm))
    graph.add_node("writer", make_writer_node(zotero_client))

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        _route_from_router,
        {
            "search": "paper_finder",
            "summarize": "summarization",
            "paper_finder": "paper_finder",
            "writer": "writer",
            "general": END,
        },
    )

    graph.add_conditional_edges(
        "paper_finder",
        _route_from_paper_finder,
        {
            "summarization": "summarization",
            END: END,
        },
    )

    # summarization always goes to END — save confirmation happens next turn
    graph.add_edge("summarization", END)
    graph.add_edge("writer", END)

    return graph.compile()
