import operator
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph

from zori.ingestion.zotero import ZoteroClient
from zori.retrieval.lexical import LexicalIndex
from zori.retrieval.search import SearchResult, SearchService


class ZoriState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    query: str
    intent: str                        # "search" | "summarize" | "general"
    target_key: str | None             # Zotero item key, set before summarization
    candidate_key: str | None          # used by display layer for Zotero links on confirmations
    display_query: str                  # cleaned query shown in search results header
    search_results: list[SearchResult]
    summary: str | None
    response: str | None


def _route_from_router(state: ZoriState) -> str:
    intent = state["intent"]
    if intent == "search":
        return "paper_finder"
    if intent == "summarize":
        return "paper_finder" if not state.get("target_key") else "summarization"
    return END  # general


def _route_from_paper_finder(state: ZoriState) -> str:
    if state.get("target_key") and state.get("intent") == "summarize":
        return "summarization"
    return END


def build_graph(
    search_service: SearchService,
    zotero_client: ZoteroClient,
    llm: BaseChatModel,
    lexical_index: LexicalIndex | None = None,
):
    from zori.agents.paper_finder import make_paper_finder_node
    from zori.agents.router import make_router_node
    from zori.agents.summarization import make_summarization_node

    graph = StateGraph(ZoriState)

    graph.add_node("router", make_router_node(llm))
    graph.add_node("paper_finder", make_paper_finder_node(search_service, llm))
    graph.add_node("summarization", make_summarization_node(zotero_client, llm, lexical_index=lexical_index))

    graph.set_entry_point("router")

    graph.add_conditional_edges("router", _route_from_router)

    graph.add_conditional_edges(
        "paper_finder",
        _route_from_paper_finder,
        {
            "summarization": "summarization",
            END: END,
        },
    )

    graph.add_edge("summarization", END)

    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("zori.retrieval.search", "SearchResult")]
    )
    return graph.compile(checkpointer=MemorySaver(serde=serde))
