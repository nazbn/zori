import pytest

from zori.agents.confirm import is_no, is_yes, redirect_to_router


@pytest.mark.parametrize("reply", ["yes", "Y", " yes ", "Yes!", "sure", "ok."])
def test_is_yes(reply):
    assert is_yes(reply)
    assert not is_no(reply)


@pytest.mark.parametrize("reply", ["no", "N", "none", "Nope.", "cancel", "no thanks"])
def test_is_no(reply):
    assert is_no(reply)
    assert not is_yes(reply)


@pytest.mark.parametrize("reply", ["yes but shorter", "no, the other one", "2", ""])
def test_other_replies_are_neither(reply):
    assert not is_yes(reply)
    assert not is_no(reply)


def test_redirect_to_router_adds_question_and_reply():
    cmd = redirect_to_router("Shall I summarize this one?", "the BERT paper instead")
    assert cmd.goto == "router"
    assert cmd.update["query"] == "the BERT paper instead"
    question, reply = cmd.update["messages"]
    assert question.type == "ai" and question.content == "Shall I summarize this one?"
    assert reply.type == "human" and reply.content == "the BERT paper instead"
