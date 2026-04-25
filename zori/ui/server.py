import logging
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
from pydantic import BaseModel

from zori.agents.graph import build_graph
from zori.cli import _fresh_state, _init_services
from zori.display.markdown import render_response_md
from zori.llm.providers import get_llm

logger = logging.getLogger(__name__)

_STATIC = Path(__file__).parent / "static"
_STATE_FILE = Path(".zori/state.json")

# ---------------------------------------------------------------------------
# Session store — maps session_id → ZoriState dict
# ---------------------------------------------------------------------------

_sessions: dict[str, dict] = {}


def _get_or_create(session_id: str | None) -> tuple[str, dict]:
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]
    sid = str(uuid.uuid4())
    _sessions[sid] = _fresh_state()
    return sid, _sessions[sid]


# ---------------------------------------------------------------------------
# App + shared state
# ---------------------------------------------------------------------------

app = FastAPI(title="Zori")

_graph = None
_services = None
_ingest_status: dict = {"state": "idle", "error": None, "papers": [], "total": 0}


@app.on_event("startup")
def _startup():
    global _graph, _services
    try:
        services, config = _init_services()
    except (FileNotFoundError, ValueError) as e:
        raise RuntimeError(f"Setup error: {e}. Run 'zori init' and configure your credentials.") from e
    _services = services
    llm = get_llm(config)
    _graph = build_graph(services.search_service, services.zotero, llm, services.lexical_index)
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

    session_id, state = _get_or_create(req.session_id)
    state["query"] = req.message.strip()
    state["messages"] = state["messages"] + [HumanMessage(content=req.message.strip())]

    try:
        state = _graph.invoke(state)
    except Exception as e:
        logger.exception("Graph error")
        raise HTTPException(status_code=500, detail=str(e))

    _sessions[session_id] = state
    return ChatResponse(session_id=session_id, response=render_response_md(state))


@app.post("/api/new-session", response_model=NewSessionResponse)
def new_session(req: ChatRequest):
    sid = str(uuid.uuid4())
    _sessions[sid] = _fresh_state()
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

def launch(host: str = "127.0.0.1", port: int = 7860, open_browser: bool = True):
    import webbrowser
    import uvicorn

    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    print(f"Zori UI → {url}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
