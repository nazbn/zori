import logging
from typing import Callable, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel

from zori.agents.graph import ZoriState
from zori.retrieval.search import SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Zori, a research assistant for a personal Zotero library.
You help researchers search, summarize, and explore their papers.

Classify the user's message into one of three intents:

- "search": user wants to find papers in their library — any phrasing like "find", "give me papers on",
  "show me", "what do I have on", "papers about", or a topic/keyword without a clear action
- "summarize": user wants a summary of a specific paper
- "general": ONLY for questions that cannot be answered by searching the library — e.g. "what does
  RAG stand for?", "explain transformers", "how does Zori work?". If in doubt, prefer "search".

For "summarize": if previous search results are provided and the user refers to one of them
(e.g. "the first one", "the attention paper", "that last result"), extract its item_key.
Otherwise set target_key to null.

For "general": provide a concise, research-focused answer in the response field.
Decline politely if the question is not research-related.

Respond only with the structured output — no extra text."""


class RouterOutput(BaseModel):
    intent: Literal["search", "summarize", "general"]
    target_key: str | None = None
    response: str | None = None


def _format_results_for_prompt(results: list[SearchResult]) -> str:
    if not results:
        return ""
    lines = ["Recent search results:"]
    for i, r in enumerate(results, 1):
        authors = ", ".join(r.authors[:2]) + (" et al." if len(r.authors) > 2 else "")
        lines.append(f"  {i}. [{r.item_key}] {r.title} — {authors} ({r.year or '?'})")
    return "\n".join(lines)


def make_router_node(llm: BaseChatModel) -> Callable[[ZoriState], dict]:
    structured_llm = llm.with_structured_output(RouterOutput)

    def router_node(state: ZoriState) -> dict:
        logger.debug(
            "[router] query=%r pending=%s confirmation_type=%r target_key=%r",
            state.get("query"), state.get("pending_confirmation"),
            state.get("confirmation_type"), state.get("target_key"),
        )
        # If a confirmation is pending (yes/no reply), skip LLM classification —
        # routing is handled by _route_from_router based on confirmation_type.
        if state.get("pending_confirmation"):
            return {}

        results_context = _format_results_for_prompt(state.get("search_results", []))
        system_content = SYSTEM_PROMPT
        if results_context:
            system_content += f"\n\n{results_context}"

        messages = [SystemMessage(content=system_content)] + state["messages"]
        output: RouterOutput = structured_llm.invoke(messages)

        updates: dict = {"intent": output.intent}

        if output.intent == "summarize":
            updates["target_key"] = output.target_key

        if output.intent == "general" and output.response:
            updates["response"] = output.response
            updates["messages"] = [AIMessage(content=output.response)]

        logger.debug(
            "[router] → intent=%r target_key=%r",
            updates.get("intent"), updates.get("target_key"),
        )
        return updates

    return router_node
