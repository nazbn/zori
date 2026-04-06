import logging
from typing import Callable

from langchain_core.messages import AIMessage
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

from zori.agents.graph import ZoriState
from zori.retrieval.formatting import format_authors, format_results, zotero_link
from zori.retrieval.search import SearchResult, SearchService

MAX_DISPLAY = 5


# ---------------------------------------------------------------------------
# LLM output schema
# ---------------------------------------------------------------------------

class SearchPlan(BaseModel):
    """How to search the library for the user's query."""
    display_query: str
    title: str | None = None            # set only for specific paper/acronym/tool names
    author: str | None = None
    year: str | None = None
    tags: list[str] | None = None
    lexical_query: str | None = None    # BM25 on papers_fts + chunks_fts
    semantic_query: str | None = None   # ChromaDB vector search

    @field_validator("title", "author", "year", "lexical_query", "semantic_query", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        return None if v == "" else v

    @field_validator("tags", mode="before")
    @classmethod
    def empty_list_to_none(cls, v):
        return None if v == [] else v


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

_QUERY_ANALYZER_PROMPT = """\
You help find papers in a personal research library.
Analyze the query and fill the search plan fields:

- display_query: the core topic, paper name, or author name (cleaned up, never filler words)
- title: the name of a specific paper, system, or acronym the user is asking for by name
- author: a researcher's name the user wants to filter by
- year: a publication year the user wants to filter by
- tags: domain keywords the user wants to filter by
- lexical_query: keyword query for BM25 search — use when the query contains specific \
technical terms, method names, or acronyms that should appear in the text
- semantic_query: query for embedding search — use when the user is asking about a \
topic or concept that requires understanding meaning, not just keyword matching

The key principle: structured fields (author, year, tags, title) are filters — use them \
whenever the user's intent is to narrow by a known attribute. \
lexical_query and semantic_query are content searches — use them when the user wants \
to find papers about a subject. A query can use filters only, content search only, or both.

When in doubt, fill in more fields rather than fewer. Redundant signals improve recall; \
missing signals lose results entirely.

Query: {query}"""


def make_paper_finder_node(
    search_service: SearchService,
    llm,
) -> Callable[[ZoriState], dict]:
    query_analyzer = llm.with_structured_output(SearchPlan)

    def paper_finder_node(state: ZoriState) -> dict:
        if state.get("pending_confirmation"):
            return _handle_confirmation(state)

        query = state["query"]
        mode = state.get("search_mode", "display")

        plan: SearchPlan = query_analyzer.invoke(
            _QUERY_ANALYZER_PROMPT.format(query=query)
        )

        logger.debug(
            "SearchPlan: display_query=%r title=%r author=%r year=%r tags=%r "
            "lexical_query=%r semantic_query=%r",
            plan.display_query, plan.title, plan.author, plan.year, plan.tags,
            plan.lexical_query, plan.semantic_query,
        )

        results = search_service.hybrid_search(
            lexical_query=plan.lexical_query,
            semantic_query=plan.semantic_query,
            title=plan.title,
            author=plan.author,
            year=plan.year,
            tags=plan.tags,
        )

        final = _group_by_paper(results)[:MAX_DISPLAY]

        # ---- Display mode (intent = "search") ----
        if mode == "display":
            response = format_results(plan.display_query, final)
            return {
                "search_results": final,
                "response": response,
                "messages": [AIMessage(content=response)],
            }

        # ---- Find-for-summarize mode (intent = "summarize", no key yet) ----
        if not final:
            response = f'I couldn\'t find "{plan.display_query}" in your library.'
            return {
                "response": response,
                "messages": [AIMessage(content=response)],
            }

        if len(final) == 1:
            top = final[0]
            authors_str = format_authors(top.authors)
            link = zotero_link(top.item_key)
            response = (
                f'I found "{top.title}" by {authors_str} ({top.year or "?"}). {link}\n'
                "Shall I summarize this one? (yes / no)"
            )
            return {
                "search_results": final,
                "pending_confirmation": True,
                "confirmation_type": "paper_selection",
                "candidate_key": top.item_key,
                "response": response,
                "messages": [AIMessage(content=response)],
            }

        # Multiple plausible results — ask the user to pick
        lines = [f'I found several papers that could match "{plan.display_query}". Which one?\n']
        for i, r in enumerate(final[:3], 1):
            authors_str = format_authors(r.authors)
            lines.append(
                f"  {i}. {r.title} — {authors_str} ({r.year or '?'}) {zotero_link(r.item_key)}"
            )
        lines.append("\nReply with a number, or 'none' to cancel.")
        response = "\n".join(lines)

        return {
            "search_results": final,
            "pending_confirmation": True,
            "confirmation_type": "paper_selection",
            "candidate_key": None,
            "response": response,
            "messages": [AIMessage(content=response)],
        }

    return paper_finder_node


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def _group_by_paper(results: list[SearchResult]) -> list[SearchResult]:
    """Keep the highest-scoring chunk per paper."""
    seen: dict[str, SearchResult] = {}
    for r in results:
        if r.item_key not in seen or r.score > seen[r.item_key].score:
            seen[r.item_key] = r
    return sorted(seen.values(), key=lambda r: r.score, reverse=True)


# ---------------------------------------------------------------------------
# Confirmation handler
# ---------------------------------------------------------------------------

def _handle_confirmation(state: ZoriState) -> dict:
    """Process the user's yes/no or numbered reply for paper selection."""
    reply = state["query"].strip().lower()
    results = state.get("search_results", [])

    if reply.isdigit():
        idx = int(reply) - 1
        if 0 <= idx < len(results):
            return {
                "target_key": results[idx].item_key,
                "pending_confirmation": False,
                "confirmation_type": None,
                "candidate_key": None,
            }

    if reply in ("yes", "y"):
        return {
            "target_key": state.get("candidate_key"),
            "pending_confirmation": False,
            "confirmation_type": None,
            "candidate_key": None,
        }

    response = "No problem. What else can I help you with?"
    return {
        "target_key": None,
        "pending_confirmation": False,
        "confirmation_type": None,
        "candidate_key": None,
        "response": response,
        "messages": [AIMessage(content=response)],
    }
