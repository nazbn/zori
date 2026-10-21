import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from zori.ingestion.zotero import ZoteroClient
from zori.retrieval.lexical import LexicalIndex
from zori.retrieval.metadata import MetadataStore
from zori.retrieval.search import SearchResult, SearchService


class ZoriState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    query: str
    intent: str                        # "search" | "summarize" | "collection" | "general"
    target_key: str | None             # Zotero item key, set before summarization
    candidate_key: str | None          # used by display layer for Zotero links on confirmations
    display_query: str                  # cleaned query shown in search results header
    search_results: list[SearchResult]
    candidates: list[SearchResult]     # papers offered for summarization, set by paper_finder
    summary: dict | None               # SummaryOutput.model_dump(), set by summarization
    summary_text: str | None           # formatted summary shown in the save confirmation
    collection_messages: Annotated[list[BaseMessage], add_messages]  # collection tool calls/results, this turn only
    response: str | None


# Literal return types let LangGraph draw the graph (get_graph / Studio) — it can't see inside the function.
def _route_from_router(
    state: ZoriState,
) -> Literal["paper_finder", "summarization", "collection_agent", "__end__"]:
    intent = state["intent"]
    if intent == "search":
        return "paper_finder"
    if intent == "summarize":
        return "paper_finder" if not state.get("target_key") else "summarization"
    if intent == "collection":
        return "collection_agent"
    return END  # general


def _route_from_paper_finder(state: ZoriState) -> Literal["confirm_paper", "__end__"]:
    # paper_finder always sets candidates in summarize mode ([] when nothing matched)
    if state.get("intent") == "summarize" and state.get("candidates"):
        return "confirm_paper"
    return END


def _route_from_summarization(state: ZoriState) -> Literal["confirm_save", "__end__"]:
    # summarization sets summary=None on every error path, so a stale summary never routes here
    return "confirm_save" if state.get("summary") else END


def build_graph(
    search_service: SearchService,
    zotero_client: ZoteroClient,
    llm: BaseChatModel,
    lexical_index: LexicalIndex | None = None,
    metadata_store: MetadataStore | None = None,
):
    from zori.agents.collection_manager import (
        collection_respond_node,
        make_collection_agent_node,
        make_collection_tools,
        make_confirm_collection_node,
        route_collection_agent,
        tool_error_message,
    )
    from zori.agents.paper_finder import confirm_paper_node, make_paper_finder_node
    from zori.agents.router import make_router_node
    from zori.agents.summarization import make_confirm_save_node, make_summarization_node

    graph = StateGraph(ZoriState)
    collection_tools = make_collection_tools(zotero_client)

    graph.add_node("router", make_router_node(llm))
    graph.add_node("paper_finder", make_paper_finder_node(search_service, llm))
    graph.add_node("confirm_paper", confirm_paper_node)
    graph.add_node("summarization", make_summarization_node(
        zotero_client, llm, metadata_store=metadata_store, lexical_index=lexical_index,
    ))
    graph.add_node("confirm_save", make_confirm_save_node(zotero_client))
    graph.add_node("collection_agent", make_collection_agent_node(llm, collection_tools))
    graph.add_node("confirm_collection", make_confirm_collection_node(zotero_client))
    graph.add_node("collection_tools", ToolNode(
        collection_tools, messages_key="collection_messages", handle_tool_errors=tool_error_message,
    ))
    graph.add_node("collection_respond", collection_respond_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges("router", _route_from_router)

    graph.add_conditional_edges(
        "paper_finder",
        _route_from_paper_finder,
        {
            "confirm_paper": "confirm_paper",
            END: END,
        },
    )
    # confirm_paper and confirm_save have no outgoing edges: they route with Command(goto=...)
    # to summarization / router / END (see zori/agents/confirm.py)

    graph.add_conditional_edges(
        "summarization",
        _route_from_summarization,
        {
            "confirm_save": "confirm_save",
            END: END,
        },
    )

    # confirm_collection also routes with Command(goto=...): collection_tools / router / END
    graph.add_conditional_edges("collection_agent", route_collection_agent)
    graph.add_edge("collection_tools", "collection_respond")
    graph.add_edge("collection_respond", END)

    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("zori.retrieval.search", "SearchResult")]
    )
    return graph.compile(checkpointer=MemorySaver(serde=serde))
