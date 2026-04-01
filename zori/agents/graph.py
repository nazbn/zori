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
    target_key: str | None             # Zotero item key for summarize requests
    save_to_zotero: bool
    search_results: list[SearchResult]
    summary: str | None
    response: str | None


def _route_to_agent(state: ZoriState) -> str:
    return state["intent"]


def _should_save(state: ZoriState) -> str:
    return "writer" if state["save_to_zotero"] else END


def build_graph(
    search_service: SearchService,
    zotero_client: ZoteroClient,
    llm: BaseChatModel,
):
    from zori.agents.retrieval import make_retrieval_node
    from zori.agents.router import make_router_node
    from zori.agents.summarization import make_summarization_node
    from zori.agents.writer import make_writer_node

    graph = StateGraph(ZoriState)

    graph.add_node("router", make_router_node(llm))
    graph.add_node("retrieval", make_retrieval_node(search_service))
    graph.add_node("summarization", make_summarization_node(zotero_client, llm))
    graph.add_node("writer", make_writer_node(zotero_client))

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        _route_to_agent,
        {
            "search": "retrieval",
            "summarize": "summarization",
            "general": END,
        },
    )

    graph.add_edge("retrieval", END)

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
