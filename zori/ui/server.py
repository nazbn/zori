import structlog
import threading
import uuid
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="chromadb")

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel

from zori.agents.graph import build_graph
from zori.cli import _init_services
from zori.display.markdown import render_interrupt_md, render_response_md
from zori.llm.providers import get_llm

logger = structlog.get_logger()

_STATIC = Path(__file__).parent / "static"
_STATE_FILE = Path(".zori/state.json")

# ---------------------------------------------------------------------------
# Session store — maps session_id → thread_id (graph state managed by MemorySaver)
# ---------------------------------------------------------------------------

_sessions: dict[str, str] = {}


def _get_or_create(session_id: str | None) -> tuple[str, str]:
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]
    sid = str(uuid.uuid4())
    _sessions[sid] = str(uuid.uuid4())
    return sid, _sessions[sid]


# ---------------------------------------------------------------------------
# App + shared state
# ---------------------------------------------------------------------------

app = FastAPI(title="Zori")

_graph = None
_services = None
_ingest_status: dict = {"state": "idle", "error": None, "papers": [], "total": 0}


_init_result = None


@app.on_event("startup")
def _startup():
    global _graph, _services
    services, config = _init_result  # pre-validated in launch()
    _services = services
    llm = get_llm(config)
    _graph = build_graph(
        services.search_service, services.zotero, llm, services.lexical_index, services.metadata_store,
    )
    logger.info("Zori graph ready.")


# ---------------------------------------------------------------------------
# API — chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str


class NewSessionResponse(BaseModel):
    session_id: str


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id, thread_id = _get_or_create(req.session_id)
    config = {"configurable": {"thread_id": thread_id}}
    message = req.message.strip()

    try:
        # Resume only if a question is pending. snapshot.next alone isn't enough: a node
        # that crashed also leaves next set, with no question for this reply to answer.
        if _graph.get_state(config).interrupts:
            result = _graph.invoke(Command(resume=message), config=config)
        else:
            result = _graph.invoke(
                {"messages": [HumanMessage(content=message)], "query": message},
                config=config,
            )
    except Exception as e:
        logger.exception("Graph error")
        raise HTTPException(status_code=500, detail=str(e))

    if "__interrupt__" in result:
        response_text = render_interrupt_md(result["__interrupt__"][0].value)
    else:
        response_text = render_response_md(result)

    return ChatResponse(session_id=session_id, response=response_text)


@app.post("/api/new-session", response_model=NewSessionResponse)
def new_session(req: ChatRequest):
    sid = str(uuid.uuid4())
    _sessions[sid] = str(uuid.uuid4())
    return NewSessionResponse(session_id=sid)


# ---------------------------------------------------------------------------
# API — ingest
# ---------------------------------------------------------------------------

class IngestStatusResponse(BaseModel):
    state: str   # idle | running | done | error
    error: str | None = None


@app.get("/api/ingest/status", response_model=IngestStatusResponse)
def ingest_status():
    ingested = _STATE_FILE.exists()
    if _ingest_status["state"] == "idle" and ingested:
        return IngestStatusResponse(state="done")
    return IngestStatusResponse(**_ingest_status)


@app.get("/api/ingest/progress")
def ingest_progress():
    return {
        "state": _ingest_status["state"],
        "papers": _ingest_status["papers"],
        "error": _ingest_status["error"],
    }


@app.post("/api/ingest")
def start_ingest():
    if _ingest_status["state"] == "running":
        raise HTTPException(status_code=409, detail="Ingestion already running.")

    def _run():
        _ingest_status["state"] = "running"
        _ingest_status["error"] = None
        _ingest_status["papers"] = []

        def _on_progress(title: str, status: str):
            _ingest_status["papers"].append({"title": title, "status": status})

        try:
            _services.pipeline.run_full(on_progress=_on_progress)
            _ingest_status["state"] = "done"
        except Exception as e:
            logger.exception("Ingestion error")
            _ingest_status["state"] = "error"
            _ingest_status["error"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/")
def index():
    return FileResponse(_STATIC / "index.html")


# ---------------------------------------------------------------------------
# Launch helper (called by cli.py)
# ---------------------------------------------------------------------------

def launch(host: str = "127.0.0.1", port: int = 7860, open_browser: bool = True, debug: bool = False):
    from zori.log import configure_logging
    configure_logging(debug=debug)
    import webbrowser
    import uvicorn

    # Fail fast with a clear message before starting the server
    global _init_result
    try:
        _init_result = _init_services()
    except (FileNotFoundError, ValueError) as e:
        print(f"\nSetup error: {e}")
        print("Run 'zori init' to create your config files, then configure your credentials.\n")
        return

    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    print(f"Zori UI → {url}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
