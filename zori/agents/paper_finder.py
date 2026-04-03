from typing import Callable, Literal

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from zori.agents.graph import ZoriState
from zori.agents.retrieval import format_authors, format_results, zotero_link
from zori.retrieval.search import SearchResult, SearchService

TOP_K = 10
MAX_DISPLAY = 5


# ---------------------------------------------------------------------------
# LLM output schemas
# ---------------------------------------------------------------------------

class SearchPlan(BaseModel):
    """How to search the library for the user's query."""
    strategies: list[Literal["concept", "title", "author", "tag"]]
    concept_query: str
    title_hint: str | None = None
    author_hint: str | None = None
    tag_hint: str | None = None


class ValidationResult(BaseModel):
    """Whether the retrieved results answer the user's query."""
    verdict: Literal["good", "partial", "none"]
    follow_up_strategy: Literal["concept", "title", "author", "tag"] | None = None
    follow_up_query: str | None = None


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def make_paper_finder_node(
    search_service: SearchService,
    llm,
    max_iterations: int = 3,
) -> Callable[[ZoriState], dict]:
    query_analyzer = llm.with_structured_output(SearchPlan)
    result_validator = llm.with_structured_output(ValidationResult)

    def paper_finder_node(state: ZoriState) -> dict:
        if state.get("pending_confirmation"):
            return _handle_confirmation(state)

        query = state["query"]
        mode = state.get("search_mode", "display")

        # ---- LLM Call 1: understand the query ----
        plan: SearchPlan = query_analyzer.invoke(
            "You help find papers in a personal research library. "
            "Analyze the query and choose the best search strategy.\n\n"
            f"Query: {query}\n\n"
            "Strategies:\n"
            '- "title": the query contains a paper title, acronym (e.g. DEM, TEASER, CNN), '
            "or named tool/system/dataset — use this aggressively for any recognizable proper noun\n"
            '- "tag": the query contains a domain keyword or tag (e.g. "urban energy", "super-resolution")\n'
            '- "author": the query mentions a person\'s name\n'
            '- "concept": broad semantic topic search — use only when no specific name/acronym is present\n\n'
            "Prefer title or tag over concept whenever the query contains a recognizable term. "
            "Always provide concept_query. Fill title_hint, author_hint, or tag_hint only when using those strategies."
        )

        # Use the cleaned query extracted by the LLM for display and follow-up searches.
        # Falls back to the raw user query only if concept_query is empty.
        display_query = plan.concept_query or query

        # ---- Retrieve ----
        results = _execute_plan(plan, search_service)

        # ---- LLM Call 2 + optional refinement loop ----
        for _ in range(max_iterations - 1):
            grouped = _group_by_paper(results)[:MAX_DISPLAY]
            if not grouped:
                break

            validation: ValidationResult = result_validator.invoke(
                f"The user searched for: {query}\n\n"
                "Retrieved papers:\n"
                + "\n".join(
                    f"- [{r.item_key}] {r.title} ({r.year or '?'}): "
                    f"{r.text[:120].replace(chr(10), ' ')}"
                    for r in grouped
                )
                + "\n\nFor each paper, decide if it is genuinely about the query topic — "
                "not just superficially similar. "
                "Set verdict to 'none' if most results are off-topic and you have a better search idea. "
                "Set verdict to 'partial' if some are relevant but a follow-up could surface more. "
                "Set verdict to 'good' only if the majority are truly on-topic. "
                "If suggesting a follow-up, prefer 'title' strategy for acronyms or specific tool names, "
                "'tag' for keywords, 'concept' for broader topics."
            )

            if validation.verdict == "good":
                break

            if validation.follow_up_strategy and validation.follow_up_query:
                more = _execute_single(
                    validation.follow_up_strategy,
                    validation.follow_up_query,
                    search_service,
                )
                if validation.verdict == "none":
                    results = more  # current results are garbage — replace entirely
                else:
                    results = _merge(results, more)
            else:
                break

        final = _group_by_paper(results)[:MAX_DISPLAY]

        # ---- Display mode (intent = "search") ----
        if mode == "display":
            response = format_results(display_query, final)
            return {
                "search_results": final,
                "response": response,
                "messages": [AIMessage(content=response)],
            }

        # ---- Find-for-summarize mode (intent = "summarize", no key yet) ----
        if not final:
            response = f'I couldn\'t find "{display_query}" in your library.'
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
        lines = [f'I found several papers that could match "{display_query}". Which one?\n']
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

def _execute_plan(plan: SearchPlan, search_service: SearchService) -> list[SearchResult]:
    # Run metadata strategies first (title / author / tag).
    # If any of them return results, skip concept — an exact metadata match
    # is always more precise than semantic similarity.
    metadata: list[SearchResult] = []
    for strategy in plan.strategies:
        if strategy != "concept":
            metadata = _merge(metadata, _execute_single(strategy, _hint(plan, strategy), search_service))

    if metadata:
        return metadata

    # Fallback: concept (semantic) search
    return _execute_single("concept", plan.concept_query, search_service)


def _hint(plan: SearchPlan, strategy: str) -> str:
    if strategy == "title" and plan.title_hint:
        return plan.title_hint
    if strategy == "author" and plan.author_hint:
        return plan.author_hint
    if strategy == "tag" and plan.tag_hint:
        return plan.tag_hint
    return plan.concept_query


def _execute_single(strategy: str, query: str, search_service: SearchService) -> list[SearchResult]:
    if strategy == "title":
        return search_service.title_search(query, top_k=TOP_K)
    if strategy == "author":
        return search_service.author_search(query, top_k=TOP_K)
    if strategy == "tag":
        return search_service.tag_search(query, top_k=TOP_K)
    return search_service.vector_search(query, top_k=TOP_K)


def _group_by_paper(results: list[SearchResult]) -> list[SearchResult]:
    """Keep the highest-scoring chunk per paper."""
    seen: dict[str, SearchResult] = {}
    for r in results:
        if r.item_key not in seen or r.score > seen[r.item_key].score:
            seen[r.item_key] = r
    return sorted(seen.values(), key=lambda r: r.score, reverse=True)


def _merge(base: list[SearchResult], new: list[SearchResult]) -> list[SearchResult]:
    """Combine two result lists, keeping the highest score per item_key."""
    combined = {r.item_key: r for r in base}
    for r in new:
        if r.item_key not in combined or r.score > combined[r.item_key].score:
            combined[r.item_key] = r
    return list(combined.values())


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
