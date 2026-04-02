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
    save_to_zotero: bool
    search_results: list[SearchResult]
    summary: str | None
    response: str | None
    # Human-in-the-loop confirmation (used by paper_finder)
    pending_confirmation: bool
    candidate_key: str | None          # proposed key awaiting user confirmation


def _route_to_agent(state: ZoriState) -> str:
    if state.get("pending_confirmation"):
        return "paper_finder"
    return state["intent"]


def _route_from_router(state: ZoriState) -> str:
    intent = state["intent"]
    if intent == "summarize" and not state.get("target_key"):
        return "paper_finder"
    return intent


def _should_save(state: ZoriState) -> str:
    return "writer" if state.get("save_to_zotero") else END


def build_graph(
    search_service: SearchService,
    zotero_client: ZoteroClient,
    llm: BaseChatModel,
):
    from zori.agents.paper_finder import make_paper_finder_node
    from zori.agents.router import make_router_node
    from zori.agents.summarization import make_summarization_node
    from zori.agents.writer import make_writer_node

    graph = StateGraph(ZoriState)

    graph.add_node("router", make_router_node(llm))
    graph.add_node("paper_finder", make_paper_finder_node(search_service))
    graph.add_node("summarization", make_summarization_node(zotero_client, llm))
    graph.add_node("writer", make_writer_node(zotero_client))

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        _route_from_router,
        {
            "search": "paper_finder",
            "summarize": "summarization",   # only reached when target_key is known
            "paper_finder": "paper_finder",
            "general": END,
        },
    )

    graph.add_conditional_edges(
        "paper_finder",
        lambda s: "summarization" if s.get("target_key") and s["intent"] == "summarize" else END,
        {
            "summarization": "summarization",
            END: END,
        },
    )

    graph.add_conditional_edges(
        "summarization",
        _should_save,
        {
            "writer": "writer",
            END: END,
        },
    )

    graph.add_edge("writer", END)

    return graph.compile()
