import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="chromadb")

import gradio as gr
from langchain_core.messages import HumanMessage

from zori.agents.graph import build_graph
from zori.cli import _fresh_state, _init_services
from zori.display.markdown import render_response_md
from zori.llm.providers import get_llm


def _build_app():
    try:
        services, config = _init_services()
    except (FileNotFoundError, ValueError) as e:
        raise RuntimeError(f"Setup error: {e}. Run 'zori init' and configure your credentials.") from e

    llm = get_llm(config)
    graph = build_graph(services.search_service, services.zotero, llm, services.lexical_index)

    def respond(message: str, history: list, zori_state: dict):
        message = message.strip()
        if not message:
            return "", history, zori_state

        zori_state["query"] = message
        zori_state["messages"] = zori_state["messages"] + [HumanMessage(content=message)]

        try:
            zori_state = graph.invoke(zori_state)
        except Exception as e:
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"Something went wrong: {e}"},
            ]
            return "", history, zori_state

        response = render_response_md(zori_state)
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]
        return "", history, zori_state

    def new_session():
        return [], _fresh_state()

    with gr.Blocks(title="Zori", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Zori\nYour multi-agent research assistant for Zotero.")

        chatbot = gr.Chatbot(height=500, show_label=False)
        state = gr.State(_fresh_state())

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Search papers, ask for a summary...",
                show_label=False,
                scale=9,
            )
            submit = gr.Button("Send", scale=1, variant="primary")

        clear = gr.Button("New Session", size="sm")

        submit.click(respond, [msg, chatbot, state], [msg, chatbot, state])
        msg.submit(respond, [msg, chatbot, state], [msg, chatbot, state])
        clear.click(new_session, None, [chatbot, state])

    return demo


def launch():
    """Launch the Zori Gradio UI."""
    _build_app().launch()
