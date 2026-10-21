"""Shared handling of user replies to interrupt() confirmations.

Confirmation nodes route with Command(goto=...) instead of outgoing edges: a node that returns
Command(goto=...) AND has outgoing edges follows both, so these nodes have no edges of their own.
"""
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

_YES = {"yes", "y", "yeah", "yep", "sure", "ok", "okay"}
_NO = {"no", "n", "nope", "none", "cancel", "no thanks"}


def is_yes(reply: str) -> bool:
    return reply.strip().lower().rstrip(".!") in _YES


def is_no(reply: str) -> bool:
    return reply.strip().lower().rstrip(".!") in _NO


def redirect_to_router(question: str, reply: str) -> Command:
    """Treat a reply that isn't an answer as a new request, e.g. "not this one, the paper named X".

    The question is added to the history so the router can resolve replies like "the other one".
    """
    return Command(
        goto="router",
        update={
            "messages": [AIMessage(content=question), HumanMessage(content=reply)],
            "query": reply,
        },
    )
