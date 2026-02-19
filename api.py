"""
CoreStack Agent — FastAPI Backend
==================================

Real-time SSE + WebSocket streaming, dynamic export serving, structured
logging, and dashboard-ready endpoints.

Endpoints
---------
POST /api/query                 → submit a query (returns query_id)
GET  /api/query/{id}            → poll status / result
GET  /api/query/{id}/stream     → SSE stream (logs + result + exports)
GET  /api/query/{id}/logs       → filtered execution logs
GET  /api/queries               → list recent queries
GET  /api/exports               → list all export files (PNG/GeoJSON/TIF…)
GET  /api/exports/{file}        → serve an export file
GET  /api/exports/{file}/preview→ inline preview (base64 image / parsed JSON)
GET  /api/health                → health check
WS   /ws/query                  → WebSocket: send query → receive live stream
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import mimetypes
import os
import re
import sys
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("corestack.api")

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORTS_DIR = os.path.join(_BASE_DIR, "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="CoreStack Agent API",
    description="Backend for the CoreStack Hybrid Geospatial Agent – "
                "streams execution in real time, serves exported artefacts.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT BACKEND  (main.py + langfuse)
# ═══════════════════════════════════════════════════════════════════════════════

from main import run_hybrid_agent  # noqa: E402
from langfuse_observability import (  # noqa: E402
    generate_session_id,
    flush as lf_flush,
    shutdown as lf_shutdown,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language geospatial question")
    session_id: Optional[str] = Field(None, description="Langfuse session ID (auto-generated if omitted)")
    user_id: Optional[str] = Field(None, description="Optional user identifier for tracing")


class QueryBrief(BaseModel):
    query_id: str
    status: str
    query: str
    started_at: str
    completed_at: Optional[str] = None
    duration_ms: Optional[float] = None
    has_result: bool = False
    has_error: bool = False
    export_count: int = 0


class ExportMeta(BaseModel):
    filename: str
    type: str
    category: str
    size_bytes: int
    modified: str
    url: str


# ═══════════════════════════════════════════════════════════════════════════════
# FILE CATEGORY MAP
# ═══════════════════════════════════════════════════════════════════════════════

_EXT_CATEGORY: Dict[str, str] = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".svg": "image",
    ".tif": "raster", ".tiff": "raster",
    ".geojson": "vector", ".shp": "vector", ".gpkg": "vector", ".kml": "vector",
    ".csv": "data", ".xlsx": "data", ".json": "data", ".txt": "data",
}

_EXT_CONTENT_TYPE: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".geojson": "application/geo+json",
    ".json": "application/json",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".gpkg": "application/geopackage+sqlite3",
    ".kml": "application/vnd.google-earth.kml+xml",
}

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _scan_exports(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return metadata for every file in exports/."""
    files: List[Dict[str, Any]] = []
    if not os.path.isdir(EXPORTS_DIR):
        return files
    for fname in os.listdir(EXPORTS_DIR):
        fpath = os.path.join(EXPORTS_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        cat = _EXT_CATEGORY.get(ext, "other")
        if category and cat != category:
            continue
        st = os.stat(fpath)
        files.append({
            "filename": fname,
            "type": ext.lstrip(".") or "unknown",
            "category": cat,
            "size_bytes": st.st_size,
            "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            "url": f"/api/exports/{fname}",
        })
    files.sort(key=lambda f: f["modified"], reverse=True)
    return files


def _snapshot_exports() -> Dict[str, float]:
    """filename → mtime snapshot of exports/."""
    snap: Dict[str, float] = {}
    if os.path.isdir(EXPORTS_DIR):
        for fname in os.listdir(EXPORTS_DIR):
            fpath = os.path.join(EXPORTS_DIR, fname)
            if os.path.isfile(fpath):
                snap[fname] = os.path.getmtime(fpath)
    return snap


def _diff_exports(before: Dict[str, float], after: Dict[str, float]) -> List[Dict[str, Any]]:
    """Return metadata dicts for files that are new or modified."""
    changed: List[Dict[str, Any]] = []
    for fname, mtime in after.items():
        if fname not in before or before[fname] != mtime:
            fpath = os.path.join(EXPORTS_DIR, fname)
            if os.path.isfile(fpath):
                ext = os.path.splitext(fname)[1].lower()
                changed.append({
                    "filename": fname,
                    "type": ext.lstrip(".") or "unknown",
                    "category": _EXT_CATEGORY.get(ext, "other"),
                    "size_bytes": os.path.getsize(fpath),
                    "url": f"/api/exports/{fname}",
                })
    return changed


# ═══════════════════════════════════════════════════════════════════════════════
# STDOUT LOG INTERCEPTOR  (filters out the mega-prompt, keeps code + results)
# ═══════════════════════════════════════════════════════════════════════════════

# Phrases that only appear inside the injected prompt template — any line
# containing one of these is suppressed from the streamed logs.
_PROMPT_FINGERPRINTS: List[str] = [
    "You are a geospatial analysis agent",
    "CRITICAL UNDERSTANDING: SPATIAL vs TIMESERIES",
    "LAYER SELECTION DECISION FRAMEWORK",
    "SANDBOX CONSTRAINTS",
    "═══════════════════",
    "EXAMPLE 1:", "EXAMPLE 1b:", "EXAMPLE 2:", "EXAMPLE 3:",
    "EXAMPLE 4:", "EXAMPLE 5:", "EXAMPLE 6:", "EXAMPLE 7:",
    "EXAMPLE 8:", "EXAMPLE 9:", "EXAMPLE 10:", "EXAMPLE 11:",
    "EXAMPLE 12:", "EXAMPLE 13:", "EXAMPLE 14:", "EXAMPLE 15:",
    "MANDATORY:", "WHY SPATIAL?", "WHY NOT TIMESERIES?",
    "WHY CHANGE RASTER?", "CORRECT CHOICE:", "WRONG CHOICE:",
    "OUTPUTS (MANDATORY",
    "Query Type 1:", "Query Type 2:", "Query Type 3:",
    "Query Type 4:", "Query Type 5:", "Query Type 6:",
    "Query Type 7:", "Query Type 8:", "Query Type 9:",
    "Query Type 10:", "Query Type 11:", "Query Type 12:",
    "Query Type 13:", "Query Type 14:", "Query Type 15:",
    "Query Type 16:", "Query Type 17:",
    "CRITICAL LAYER NAME MATCHING",
    "SURFACE WATER DATA COLUMN REFERENCE",
    "ACTUAL CHANGE DETECTION LAYER NAMES",
    "osmnx, geopandas, shapely, matplotlib",
    "═══════════════════════════════════════",
    "YEAR COLUMNS:",
    "fetch_corestack_data tool FIRST",
    "Make sure to wrap your final answer",
    "final_answer(\"The final answer",
    "WHY CHANGE RASTER",
    "WHY NOT TIMESERIES",
    "NEVER use bare `open()`",
    "NEVER reproject to UTM",
]

# If a single write() call is longer than this and matches no "interesting"
# pattern it is almost certainly the prompt being echoed → suppress.
_MAX_BORING_LEN = 800


def _is_prompt_content(text: str) -> bool:
    """Return True when *text* looks like part of the injected prompt."""
    for fp in _PROMPT_FINGERPRINTS:
        if fp in text:
            return True
    return False


class _LogInterceptor:
    """
    Replaces ``sys.stdout`` while the agent runs.

    * Forwards everything to the **real** console (``_real``) so the dev
      terminal still works.
    * Filters out prompt content and pushes clean lines into a
      per-query log list for the API to stream.
    """

    def __init__(self, query_id: str, log_sink: List[str], real_stdout, user_query: str = ""):
        self._qid = query_id
        self._sink = log_sink
        self._real = real_stdout
        self._buf = ""  # partial-line buffer
        self._user_query = user_query.strip().lower()

    # ── io.TextIOBase protocol ──────────────────────────────────────────────

    def write(self, data: str) -> int:
        # Always forward to real stdout
        if self._real:
            try:
                self._real.write(data)
            except Exception:
                pass

        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._process_line(line)
        return len(data)

    def flush(self):
        if self._real:
            try:
                self._real.flush()
            except Exception:
                pass

    def fileno(self):
        if self._real:
            return self._real.fileno()
        raise io.UnsupportedOperation("fileno")

    @property
    def encoding(self):
        return getattr(self._real, "encoding", "utf-8")

    def isatty(self):
        return False

    # ── internal ────────────────────────────────────────────────────────────

    def _process_line(self, line: str):
        stripped = line.strip()
        if not stripped:
            return

        # Drop prompt content
        if _is_prompt_content(stripped):
            return

        # Drop the user's own query text from logs
        if self._user_query:
            lower = stripped.lower()
            if lower == self._user_query:
                return
            # Also catch "New task: <query>" or similar wrapper lines
            if lower.endswith(self._user_query) and len(stripped) < len(self._user_query) + 40:
                return

        # Drop very long lines that are almost certainly prompt echo
        if len(stripped) > _MAX_BORING_LEN:
            # keep JSON blobs (API responses) and tracebacks
            if not (stripped.startswith("{") or stripped.startswith("Traceback")):
                return

        self._sink.append(stripped)


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

_AGENT_LOCK = threading.Lock()  # serialise agent runs (shared stdout)


class _QueryStore:
    """Thread-safe in-memory store for the last N query records."""

    _MAX = 200

    def __init__(self):
        self._data: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def create(self, query: str, session_id: str | None, user_id: str | None) -> str:
        qid = uuid.uuid4().hex[:10]
        rec = {
            "query_id": qid,
            "status": "pending",
            "query": query,
            "session_id": session_id or generate_session_id(),
            "user_id": user_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "duration_ms": None,
            "result": None,
            "error": None,
            "code_blocks": [],
            "new_exports": [],
            "logs": [],           # filtered log lines (list[str])
        }
        with self._lock:
            self._data[qid] = rec
            while len(self._data) > self._MAX:
                self._data.popitem(last=False)
        return qid

    def get(self, qid: str) -> Dict[str, Any] | None:
        return self._data.get(qid)

    def update(self, qid: str, **kw):
        with self._lock:
            if qid in self._data:
                self._data[qid].update(kw)

    def all(self) -> List[Dict[str, Any]]:
        return list(self._data.values())


store = _QueryStore()


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND AGENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_code_blocks(text: str) -> List[str]:
    if not text:
        return []
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", str(text), flags=re.DOTALL)
    return [b.strip() for b in blocks if b.strip()]


def _run_agent(qid: str):
    """Execute the agent synchronously (called from a daemon thread)."""
    rec = store.get(qid)
    if rec is None:
        return

    store.update(qid, status="running")
    logger.info("Agent started  [%s]  %s", qid, rec["query"][:80])

    # Snapshot exports BEFORE
    snap_before = _snapshot_exports()

    t0 = time.perf_counter()
    original_stdout = sys.stdout

    with _AGENT_LOCK:
        # Install interceptor
        interceptor = _LogInterceptor(qid, rec["logs"], original_stdout, user_query=rec["query"])
        sys.stdout = interceptor  # type: ignore[assignment]

        try:
            result = run_hybrid_agent(
                user_query=rec["query"],
                exports_dir=EXPORTS_DIR,
                session_id=rec["session_id"],
                user_id=rec["user_id"],
            )

            elapsed = (time.perf_counter() - t0) * 1000
            snap_after = _snapshot_exports()

            store.update(
                qid,
                status="completed",
                result=str(result),
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=round(elapsed, 2),
                new_exports=_diff_exports(snap_before, snap_after),
                code_blocks=_extract_code_blocks(str(result)),
            )
            logger.info("Agent finished [%s]  %.1fs", qid, elapsed / 1000)

        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            store.update(
                qid,
                status="failed",
                error=str(exc),
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=round(elapsed, 2),
            )
            logger.error("Agent failed   [%s]  %s", qid, exc)

        finally:
            sys.stdout = original_stdout
            try:
                lf_flush()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# REST ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── health ──────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["system"])
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exports_dir": EXPORTS_DIR,
        "export_count": len(_scan_exports()),
    }


# ── submit query ────────────────────────────────────────────────────────────

@app.post("/api/query", tags=["query"])
async def submit_query(body: QueryRequest):
    """
    Submit a geospatial query.  Returns immediately with a ``query_id``.
    Connect to ``/api/query/{id}/stream`` for real-time SSE updates.
    """
    qid = store.create(body.query, body.session_id, body.user_id)
    logger.info("Query created   [%s]  %s", qid, body.query[:80])

    thread = threading.Thread(target=_run_agent, args=(qid,), daemon=True)
    thread.start()

    return {
        "query_id": qid,
        "status": "pending",
        "stream_url": f"/api/query/{qid}/stream",
    }


# ── poll result ─────────────────────────────────────────────────────────────

@app.get("/api/query/{qid}", tags=["query"])
async def get_query(qid: str):
    """Poll the full result of a query (use SSE stream for real-time)."""
    rec = store.get(qid)
    if rec is None:
        raise HTTPException(404, f"Query {qid} not found")

    return {
        "query_id": rec["query_id"],
        "status": rec["status"],
        "query": rec["query"],
        "started_at": rec["started_at"],
        "completed_at": rec["completed_at"],
        "duration_ms": rec["duration_ms"],
        "result": rec["result"],
        "error": rec["error"],
        "code_blocks": rec["code_blocks"],
        "new_exports": rec["new_exports"],
    }


# ── execution logs (filtered) ──────────────────────────────────────────────

@app.get("/api/query/{qid}/logs", tags=["query"])
async def get_query_logs(qid: str, since: int = 0):
    """
    Return filtered execution logs.  Pass ``since=<n>`` to fetch only new
    lines (long-polling friendly).
    """
    rec = store.get(qid)
    if rec is None:
        raise HTTPException(404, f"Query {qid} not found")

    logs: list = rec["logs"]
    return {
        "query_id": qid,
        "status": rec["status"],
        "total": len(logs),
        "logs": logs[since:],
        "next_since": len(logs),
    }


# ── SSE stream ─────────────────────────────────────────────────────────────

@app.get("/api/query/{qid}/stream", tags=["query"])
async def stream_query(qid: str):
    """
    Server-Sent Events stream.  Events:

    * ``log``        — a single filtered execution log line
    * ``status``     — running / completed / failed
    * ``result``     — final agent answer (text)
    * ``exports``    — list of new/modified export files
    * ``code``       — code blocks extracted from the answer
    * ``duration_ms``— wall-clock execution time
    * ``done``       — terminal event; close the connection
    """
    rec = store.get(qid)
    if rec is None:
        raise HTTPException(404, f"Query {qid} not found")

    def _sse(event_type: str, data: Any) -> str:
        payload = json.dumps({"type": event_type, "data": data}, default=str)
        return f"data: {payload}\n\n"

    async def _generator():
        cursor = 0

        # initial status
        yield _sse("status", store.get(qid)["status"])

        while True:
            snap = store.get(qid)
            if snap is None:
                break

            # stream new log lines
            logs: list = snap["logs"]
            if len(logs) > cursor:
                for line in logs[cursor:]:
                    yield _sse("log", line)
                cursor = len(logs)

            # terminal states
            if snap["status"] in ("completed", "failed"):
                yield _sse("status", snap["status"])

                if snap["status"] == "completed":
                    yield _sse("result", snap["result"])
                    yield _sse("exports", snap["new_exports"])
                    yield _sse("code", snap["code_blocks"])
                    yield _sse("duration_ms", snap["duration_ms"])
                else:
                    yield _sse("error", snap["error"])
                    yield _sse("duration_ms", snap["duration_ms"])

                yield _sse("done", None)
                break

            await asyncio.sleep(0.4)

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── list queries ────────────────────────────────────────────────────────────

@app.get("/api/queries", tags=["query"])
async def list_queries(status: Optional[str] = None, limit: int = 50):
    """List recent queries.  Optional filter by *status*."""
    items = store.all()
    if status:
        items = [q for q in items if q["status"] == status]

    out = []
    for q in items[-limit:]:
        out.append({
            "query_id": q["query_id"],
            "status": q["status"],
            "query": q["query"],
            "started_at": q["started_at"],
            "completed_at": q["completed_at"],
            "duration_ms": q["duration_ms"],
            "has_result": q["result"] is not None,
            "has_error": q["error"] is not None,
            "export_count": len(q.get("new_exports", [])),
        })

    return {"queries": out, "total": len(out)}


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/exports", tags=["exports"])
async def list_exports(category: Optional[str] = None):
    """
    List all files in ``exports/``.

    Optional filter: ``?category=image|vector|raster|data``
    """
    files = _scan_exports(category)
    return {"exports": files, "total": len(files)}


@app.get("/api/exports/{filename}", tags=["exports"])
async def serve_export(filename: str):
    """Download / serve an export file with the correct Content-Type."""
    fpath = os.path.join(EXPORTS_DIR, filename)
    if not os.path.isfile(fpath):
        raise HTTPException(404, f"Export not found: {filename}")

    ext = os.path.splitext(filename)[1].lower()
    media = _EXT_CONTENT_TYPE.get(ext, "application/octet-stream")
    return FileResponse(fpath, media_type=media, filename=filename)


@app.get("/api/exports/{filename}/preview", tags=["exports"])
async def preview_export(filename: str):
    """
    Inline preview suitable for a dashboard.

    * **GeoJSON / JSON** → parsed object
    * **PNG / JPG / TIF** → base64 data-URI
    * **CSV / TXT** → plain text (first 200 KB)
    """
    fpath = os.path.join(EXPORTS_DIR, filename)
    if not os.path.isfile(fpath):
        raise HTTPException(404, f"Export not found: {filename}")

    ext = os.path.splitext(filename)[1].lower()
    size = os.path.getsize(fpath)

    # ── vector / json ───────────────────────────────────────────────────────
    if ext in (".geojson", ".json"):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "type": "geojson" if ext == ".geojson" else "json",
                "filename": filename,
                "size_bytes": size,
                "data": data,
            }
        except json.JSONDecodeError:
            with open(fpath, "r", encoding="utf-8") as f:
                return {"type": "text", "filename": filename, "data": f.read(200_000)}

    # ── image / raster ──────────────────────────────────────────────────────
    if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svg"):
        with open(fpath, "rb") as f:
            raw = f.read()
        mime = _EXT_CONTENT_TYPE.get(ext, "application/octet-stream")
        b64 = base64.b64encode(raw).decode("ascii")
        return {
            "type": "image",
            "filename": filename,
            "format": ext.lstrip("."),
            "size_bytes": size,
            "data_uri": f"data:{mime};base64,{b64}",
        }

    # ── text / csv ──────────────────────────────────────────────────────────
    if ext in (".csv", ".txt"):
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(200_000)
        return {"type": "text", "filename": filename, "size_bytes": size, "data": text}

    # ── fallback ────────────────────────────────────────────────────────────
    return {
        "type": "binary",
        "filename": filename,
        "size_bytes": size,
        "message": f"No preview for .{ext.lstrip('.')} files — download via /api/exports/{filename}",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/query")
async def ws_query(ws: WebSocket):
    """
    WebSocket for bidirectional interaction::

        → {"query": "...", "session_id": "...", "user_id": "..."}
        ← {"type": "query_id",    "data": "abc123"}
        ← {"type": "log",         "data": "..."}   (repeated)
        ← {"type": "result",      "data": "..."}
        ← {"type": "exports",     "data": [...]}
        ← {"type": "code",        "data": [...]}
        ← {"type": "duration_ms", "data": 12345.6}
        ← {"type": "done",        "data": null}
    """
    await ws.accept()
    logger.info("WebSocket connected")

    try:
        while True:
            payload = await ws.receive_json()
            query_text = (payload.get("query") or "").strip()
            if not query_text:
                await ws.send_json({"type": "error", "data": "Empty query"})
                continue

            qid = store.create(
                query_text,
                payload.get("session_id"),
                payload.get("user_id"),
            )
            await ws.send_json({"type": "query_id", "data": qid})

            # launch agent
            thread = threading.Thread(target=_run_agent, args=(qid,), daemon=True)
            thread.start()

            # stream back
            cursor = 0
            while True:
                snap = store.get(qid)
                if snap is None:
                    break

                logs: list = snap["logs"]
                if len(logs) > cursor:
                    for line in logs[cursor:]:
                        await ws.send_json({"type": "log", "data": line})
                    cursor = len(logs)

                if snap["status"] in ("completed", "failed"):
                    if snap["status"] == "completed":
                        await ws.send_json({"type": "result", "data": snap["result"]})
                        await ws.send_json({"type": "exports", "data": snap.get("new_exports", [])})
                        await ws.send_json({"type": "code", "data": snap.get("code_blocks", [])})
                        await ws.send_json({"type": "duration_ms", "data": snap["duration_ms"]})
                    else:
                        await ws.send_json({"type": "error", "data": snap["error"]})
                        await ws.send_json({"type": "duration_ms", "data": snap["duration_ms"]})

                    await ws.send_json({"type": "done", "data": qid})
                    break

                await asyncio.sleep(0.4)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE HOOKS
# ═══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def _on_startup():
    n = len(_scan_exports())
    logger.info("CoreStack API ready  │  exports_dir=%s  │  %d files", EXPORTS_DIR, n)


@app.on_event("shutdown")
async def _on_shutdown():
    try:
        lf_shutdown()
    except Exception:
        pass
    logger.info("CoreStack API stopped")
