from typing import Callable, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel

from zori.agents.graph import ZoriState
from zori.retrieval.search import SearchResult

SYSTEM_PROMPT = """You are Zori, a research assistant for a personal Zotero library.
You help researchers search, summarize, and explore their papers.

Classify the user's message into one of three intents:

- "search": user wants to find papers on a topic, author, or keyword
- "summarize": user wants a summary of a specific paper
- "general": a research question you can answer directly from your knowledge

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

        return updates

    return router_node
