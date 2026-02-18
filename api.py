"""
CoreStack Hybrid Agent — FastAPI Server
========================================

Production-ready REST + WebSocket API that exposes the SmolAgents CodeAgent
pipeline and serves all generated exports (GeoJSON, PNG, TIFF, TXT).

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints overview:
───────────────────
POST   /query                  → Submit a new agent query (async background task)
GET    /query/{query_id}       → Poll query status & result
WS     /ws/query               → WebSocket for real-time streaming output
GET    /exports                → List all export files with metadata
GET    /exports/{filename}     → Download / stream a single export file
GET    /exports/{filename}/preview → Preview GeoJSON as JSON, PNG as base64
DELETE /exports/{filename}     → Delete an export file
GET    /sessions               → List Langfuse sessions
GET    /sessions/{session_id}  → Get traces for a session
POST   /feedback               → Submit user feedback (thumbs up/down)
GET    /health                 → Health check
GET    /config                 → Current configuration (non-secret)

Output types handled (based on all 17 query outputs):
─────────────────────────────────────────────────────
  .geojson  → Spatial vector data (cropping intensity, drought, deforestation …)
  .png      → Charts, scatter plots, heatmaps, bar charts
  .tif      → Raster data (LULC, change detection)
  .txt      → Execution logs / text reports
"""

from __future__ import annotations

import asyncio
import base64
import glob
import json
import mimetypes
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

# ============================================================================
# App initialization
# ============================================================================

app = FastAPI(
    title="CoreStack Hybrid Agent API",
    description=(
        "REST + WebSocket API for the SmolAgents-based geospatial analysis agent. "
        "Submit natural-language queries, poll results, stream real-time output, "
        "download exports (GeoJSON / PNG / TIFF), and provide feedback."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (allow any origin in dev; lock down in production) ──────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
EXPORTS_DIR = BASE_DIR / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)

# ── Thread pool for running the synchronous agent in a background thread ─────
_executor = ThreadPoolExecutor(max_workers=int(os.getenv("AGENT_WORKERS", "2")))


# ============================================================================
# Lazy imports (avoid heavy startup cost for health-check requests)
# ============================================================================

_main_module = None


def _get_main():
    """Import main.py lazily so that EE / Langfuse / model init happens once."""
    global _main_module
    if _main_module is None:
        import main as _m
        _main_module = _m
    return _main_module


# ============================================================================
# Pydantic models
# ============================================================================

class QueryStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryRequest(BaseModel):
    """Request body for POST /query."""
    query: str = Field(..., min_length=5, description="Natural-language question for the agent")
    session_id: Optional[str] = Field(None, description="Langfuse session ID (auto-generated if omitted)")
    user_id: Optional[str] = Field(None, description="End-user identifier for tracing")
    exports_dir: Optional[str] = Field(None, description="Custom exports directory (default: ./exports)")

    model_config = {"json_schema_extra": {
        "examples": [{
            "query": "Could you show how cropping intensity has changed over the years in Navalgund, Dharwad, Karnataka?",
            "session_id": None,
            "user_id": "demo-user",
        }]
    }}


class QueryResponse(BaseModel):
    """Status envelope returned by GET /query/{id} and POST /query."""
    query_id: str
    status: QueryStatus
    query: str
    session_id: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    duration_s: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    exports: List[str] = Field(default_factory=list, description="Files created by this run")


class FeedbackRequest(BaseModel):
    """Request body for POST /feedback."""
    query_id: Optional[str] = Field(None, description="Query ID to attach feedback to")
    trace_id: Optional[str] = Field(None, description="Langfuse trace ID (alternative to query_id)")
    score: float = Field(..., ge=0, le=1, description="Numeric score 0–1 (1 = thumbs up)")
    comment: Optional[str] = Field(None, description="Optional free-text comment")


class ExportFileInfo(BaseModel):
    """Metadata for a single export file."""
    filename: str
    extension: str
    size_bytes: int
    size_human: str
    mime_type: Optional[str]
    created_at: str
    url: str
    preview_url: Optional[str] = None


# ============================================================================
# In-memory query store  (swap for Redis / DB in production)
# ============================================================================

_query_store: Dict[str, Dict[str, Any]] = {}


def _record_query(query_id: str, query: str, session_id: str) -> Dict[str, Any]:
    entry = {
        "query_id": query_id,
        "status": QueryStatus.PENDING,
        "query": query,
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "completed_at": None,
        "duration_s": None,
        "result": None,
        "error": None,
        "exports_before": set(_list_export_filenames()),
    }
    _query_store[query_id] = entry
    return entry


# ============================================================================
# Background agent runner
# ============================================================================

def _run_agent_sync(query_id: str, query: str, session_id: str,
                    user_id: Optional[str], exports_dir: Optional[str]) -> None:
    """
    Run the hybrid agent synchronously (called inside a thread).

    Updates ``_query_store[query_id]`` with result / error / exports.
    """
    entry = _query_store[query_id]
    entry["status"] = QueryStatus.RUNNING
    t0 = time.perf_counter()

    try:
        main = _get_main()
        result = main.run_hybrid_agent(
            user_query=query,
            exports_dir=exports_dir or str(EXPORTS_DIR),
            session_id=session_id,
            user_id=user_id,
        )
        elapsed = time.perf_counter() - t0

        # Detect new files produced during this run
        exports_after = set(_list_export_filenames())
        new_files = sorted(exports_after - entry["exports_before"])

        entry.update({
            "status": QueryStatus.COMPLETED,
            "result": str(result),
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "duration_s": round(elapsed, 2),
            "exports": new_files,
        })

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        entry.update({
            "status": QueryStatus.FAILED,
            "error": str(exc),
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "duration_s": round(elapsed, 2),
        })


# ============================================================================
# Helper utilities
# ============================================================================

def _list_export_filenames() -> List[str]:
    """Return sorted list of filenames in the exports directory."""
    if not EXPORTS_DIR.exists():
        return []
    return sorted(f.name for f in EXPORTS_DIR.iterdir() if f.is_file())


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def _file_info(filepath: Path) -> ExportFileInfo:
    stat = filepath.stat()
    ext = filepath.suffix.lower()
    mime, _ = mimetypes.guess_type(filepath.name)

    # Custom MIME overrides
    if ext == ".geojson":
        mime = "application/geo+json"
    elif ext == ".tif" or ext == ".tiff":
        mime = "image/tiff"

    has_preview = ext in (".geojson", ".png", ".txt")

    return ExportFileInfo(
        filename=filepath.name,
        extension=ext,
        size_bytes=stat.st_size,
        size_human=_human_size(stat.st_size),
        mime_type=mime,
        created_at=datetime.fromtimestamp(stat.st_mtime).isoformat() + "Z",
        url=f"/exports/{filepath.name}",
        preview_url=f"/exports/{filepath.name}/preview" if has_preview else None,
    )


# ── Mapping: query number → known export files produced ──────────────────────
# This maps each of the 17 prompts to the export files it generates so clients
# can look up which outputs belong to which analysis type.

QUERY_EXPORT_MAP: Dict[int, Dict[str, Any]] = {
    1: {
        "title": "Cropping Intensity Trend",
        "description": "Cropping intensity change over years in a tehsil",
        "exports": ["cropping_intensity_over_years.png", "navalgund_cropping_intensity_trend.png"],
        "output_types": ["chart"],
    },
    2: {
        "title": "Surface Water Availability Trend",
        "description": "Surface water availability change over years",
        "exports": ["navalgund_surface_water_availability_periods.png", "navalgund_surface_water_bodies.geojson"],
        "output_types": ["chart", "geojson"],
    },
    3: {
        "title": "Tree Cover Loss / Degradation",
        "description": "Areas that lost tree cover since a given year, with degraded hectares",
        "exports": ["deforestation_navalgund.geojson", "degradation_navalgund.geojson"],
        "output_types": ["geojson"],
    },
    4: {
        "title": "Cropland to Built-up Change",
        "description": "Cropland converted to built-up areas with regions shown",
        "exports": ["cropland_to_builtup_change.png", "crop_to_builtup_change.tif", "lulc_old_test.tif", "lulc_new_test.tif"],
        "output_types": ["chart", "raster"],
    },
    5: {
        "title": "Drought Affected Villages",
        "description": "Villages that have experienced droughts",
        "exports": ["drought_affected_villages_navalgund.geojson"],
        "output_types": ["geojson"],
    },
    6: {
        "title": "Drought-Sensitive Microwatersheds (Cropping)",
        "description": "Microwatersheds with highest cropping sensitivity to drought",
        "exports": ["top_drought_sensitive_microwatersheds.geojson"],
        "output_types": ["geojson"],
    },
    7: {
        "title": "Drought-Sensitive Microwatersheds (Surface Water)",
        "description": "Microwatersheds with highest surface water sensitivity to drought",
        "exports": ["top_sw_sensitive_microwatersheds.geojson"],
        "output_types": ["geojson"],
    },
    8: {
        "title": "Similar Microwatersheds (Feature Matching)",
        "description": "MWS similar to a reference based on terrain, drought, LULC, CI",
        "exports": ["similar_microwatersheds.geojson"],
        "output_types": ["geojson"],
    },
    9: {
        "title": "Similar Microwatersheds (PSM)",
        "description": "MWS similar via propensity score matching",
        "exports": ["psm_matched_microwatersheds.geojson"],
        "output_types": ["geojson"],
    },
    10: {
        "title": "Ranked MWS by CI & Surface Water",
        "description": "Top-K drought/SW sensitive MWS ranked by cropping + water scores",
        "exports": ["ranked_mws_by_ci_and_sw.geojson"],
        "output_types": ["geojson"],
    },
    11: {
        "title": "SC/ST% vs NREGA Works Scatter",
        "description": "Village-level SC/ST population vs NREGA works scatter plot",
        "exports": ["scst_vs_nrega_scatter.png", "scst_vs_nrega_villages.geojson"],
        "output_types": ["chart", "geojson"],
    },
    12: {
        "title": "Cropping Intensity vs Runoff Quadrants",
        "description": "Four-quadrant scatter: high/low CI vs high/low runoff",
        "exports": ["ci_vs_runoff_quadrant_scatter.png", "ci_vs_runoff_quadrants.geojson", "qt12_output.txt"],
        "output_types": ["chart", "geojson", "text"],
    },
    13: {
        "title": "Temperature vs Cropping Intensity Scatter",
        "description": "Average monsoon LST vs cropping intensity scatter plot",
        "exports": ["lst_vs_ci_scatter.png", "lst_vs_cropping_intensity.geojson"],
        "output_types": ["chart", "geojson"],
    },
    14: {
        "title": "Phenological Stages",
        "description": "Regions with similar phenological cycles per month",
        "exports": ["phenological_stages_heatmap.png", "phenological_stages_navalgund.geojson"],
        "output_types": ["chart", "geojson"],
    },
    15: {
        "title": "Runoff vs CI by Phenological Stage",
        "description": "Runoff accumulation per phenostage vs cropping intensity",
        "exports": ["runoff_vs_ci_by_phenostage.png", "runoff_vs_ci_by_phenostage.geojson"],
        "output_types": ["chart", "geojson"],
    },
    16: {
        "title": "LST–CI Hypothesis Test",
        "description": "Hypothesis test: higher temp → higher cropping intensity",
        "exports": ["lst_ci_hypothesis_test.png", "lst_ci_hypothesis_test.geojson"],
        "output_types": ["chart", "geojson"],
    },
    17: {
        "title": "Agricultural Suitability Index",
        "description": "Composite ASI ranking of microwatersheds",
        "exports": ["agricultural_suitability_index_ranking.png", "agricultural_suitability_index_ranked_mws.geojson"],
        "output_types": ["chart", "geojson"],
    },
}


# ============================================================================
# ROUTES — Health & Config
# ============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    """Health check — reports service status, Langfuse connectivity, and EE init."""
    langfuse_ok = False
    try:
        from langfuse_observability import is_enabled
        langfuse_ok = is_enabled()
    except Exception:
        pass

    ee_ok = False
    try:
        import ee as _ee
        _ee.Number(1).getInfo()
        ee_ok = True
    except Exception:
        pass

    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "langfuse_connected": langfuse_ok,
        "earth_engine_connected": ee_ok,
        "exports_dir": str(EXPORTS_DIR),
        "exports_count": len(_list_export_filenames()),
    }


@app.get("/config", tags=["System"])
async def get_config():
    """Return non-secret configuration details."""
    return {
        "model_id": "gemini/gemini-2.5-flash-lite",
        "corestack_base_url": os.getenv("CORESTACK_BASE_URL", "https://geoserver.core-stack.org/api/v1"),
        "exports_dir": str(EXPORTS_DIR),
        "langfuse_host": os.getenv("LANGFUSE_HOST", os.getenv("LANGFUSE_BASE_URL", "")),
        "agent_workers": int(os.getenv("AGENT_WORKERS", "2")),
        "query_export_map": QUERY_EXPORT_MAP,
    }


# ============================================================================
# ROUTES — Query (submit + poll)
# ============================================================================

@app.post("/query", response_model=QueryResponse, status_code=202, tags=["Agent"])
async def submit_query(req: QueryRequest):
    """
    Submit a natural-language query to the hybrid agent.

    The agent runs asynchronously in a background thread. Poll
    ``GET /query/{query_id}`` for the result, or use the WebSocket
    endpoint ``/ws/query`` for real-time streaming.

    Returns 202 Accepted with a ``query_id`` for tracking.
    """
    main = _get_main()

    query_id = uuid.uuid4().hex[:12]
    session_id = req.session_id or main.generate_session_id()

    entry = _record_query(query_id, req.query, session_id)

    # Launch agent in background thread (non-blocking)
    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        _executor,
        _run_agent_sync,
        query_id,
        req.query,
        session_id,
        req.user_id,
        req.exports_dir,
    )

    return QueryResponse(
        query_id=query_id,
        status=QueryStatus.PENDING,
        query=req.query,
        session_id=session_id,
        created_at=entry["created_at"],
    )


@app.get("/query/{query_id}", response_model=QueryResponse, tags=["Agent"])
async def get_query_status(query_id: str):
    """
    Poll the status and result of a previously submitted query.

    Returns ``pending`` | ``running`` | ``completed`` | ``failed``.
    When ``completed``, the ``result`` field contains the agent's answer
    and ``exports`` lists newly created files.
    """
    entry = _query_store.get(query_id)
    if not entry:
        raise HTTPException(404, f"Query {query_id} not found")

    return QueryResponse(
        query_id=entry["query_id"],
        status=entry["status"],
        query=entry["query"],
        session_id=entry.get("session_id"),
        created_at=entry["created_at"],
        completed_at=entry.get("completed_at"),
        duration_s=entry.get("duration_s"),
        result=entry.get("result"),
        error=entry.get("error"),
        exports=entry.get("exports", []),
    )


@app.get("/queries", tags=["Agent"])
async def list_queries(
    status: Optional[QueryStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=500),
):
    """List recent queries, optionally filtered by status."""
    items = list(_query_store.values())
    if status:
        items = [i for i in items if i["status"] == status]
    items = sorted(items, key=lambda x: x["created_at"], reverse=True)[:limit]
    return {"total": len(items), "queries": items}


# ============================================================================
# ROUTES — WebSocket (real-time streaming)
# ============================================================================

@app.websocket("/ws/query")
async def websocket_query(ws: WebSocket):
    """
    WebSocket endpoint for real-time agent output streaming.

    Protocol:
    1. Client connects.
    2. Client sends JSON: ``{"query": "...", "session_id": "...", "user_id": "..."}``
    3. Server streams back JSON messages:
       - ``{"type": "status", "status": "running"}``
       - ``{"type": "log", "data": "..."}``          (stdout lines)
       - ``{"type": "result", "data": "...", "exports": [...]}``
       - ``{"type": "error", "error": "..."}``
    4. Connection closes after result / error.
    """
    await ws.accept()

    try:
        # 1. Receive the query
        raw = await ws.receive_text()
        payload = json.loads(raw)
        user_query = payload.get("query", "")
        session_id = payload.get("session_id")
        user_id = payload.get("user_id")

        if not user_query or len(user_query) < 5:
            await ws.send_json({"type": "error", "error": "Query must be at least 5 characters"})
            await ws.close()
            return

        main = _get_main()
        if not session_id:
            session_id = main.generate_session_id()

        query_id = uuid.uuid4().hex[:12]
        _record_query(query_id, user_query, session_id)

        await ws.send_json({"type": "status", "status": "running", "query_id": query_id, "session_id": session_id})

        # 2. Run agent in thread, capture stdout lines
        import io, sys
        from contextlib import redirect_stdout

        class _WSBuffer:
            """Captures stdout and asynchronously pushes lines to the WebSocket."""
            def __init__(self, ws_ref: WebSocket, loop_ref):
                self._ws = ws_ref
                self._loop = loop_ref
                self._original = sys.stdout
                self._buffer = ""

            def write(self, data: str) -> int:
                self._original.write(data)  # keep console output
                self._buffer += data
                while "\n" in self._buffer:
                    line, self._buffer = self._buffer.split("\n", 1)
                    if line.strip():
                        asyncio.run_coroutine_threadsafe(
                            self._send_line(line), self._loop
                        )
                return len(data)

            def flush(self):
                self._original.flush()

            async def _send_line(self, line: str):
                try:
                    await self._ws.send_json({"type": "log", "data": line})
                except Exception:
                    pass

        loop = asyncio.get_running_loop()

        def _ws_agent_runner():
            ws_buf = _WSBuffer(ws, loop)
            old_stdout = sys.stdout
            sys.stdout = ws_buf
            try:
                result = main.run_hybrid_agent(
                    user_query=user_query,
                    exports_dir=str(EXPORTS_DIR),
                    session_id=session_id,
                    user_id=user_id,
                )
                return result
            finally:
                sys.stdout = old_stdout

        # 3. Execute
        t0 = time.perf_counter()
        try:
            result = await loop.run_in_executor(_executor, _ws_agent_runner)
            elapsed = round(time.perf_counter() - t0, 2)

            exports_after = set(_list_export_filenames())
            exports_before = _query_store[query_id].get("exports_before", set())
            new_files = sorted(exports_after - exports_before)

            _query_store[query_id].update({
                "status": QueryStatus.COMPLETED,
                "result": str(result),
                "completed_at": datetime.utcnow().isoformat() + "Z",
                "duration_s": elapsed,
                "exports": new_files,
            })

            await ws.send_json({
                "type": "result",
                "query_id": query_id,
                "data": str(result),
                "exports": new_files,
                "duration_s": elapsed,
            })

        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 2)
            _query_store[query_id].update({
                "status": QueryStatus.FAILED,
                "error": str(exc),
                "completed_at": datetime.utcnow().isoformat() + "Z",
                "duration_s": elapsed,
            })
            await ws.send_json({"type": "error", "error": str(exc), "duration_s": elapsed})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ============================================================================
# ROUTES — Exports (list / download / preview / delete)
# ============================================================================

@app.get("/exports", tags=["Exports"])
async def list_exports(
    extension: Optional[str] = Query(None, description="Filter by extension (.geojson, .png, .tif, .txt)"),
    query_number: Optional[int] = Query(None, ge=1, le=17, description="Filter by query # (1–17)"),
):
    """
    List all exported files with metadata.

    Optionally filter by file extension or query number (1–17) to see
    which exports belong to which analysis.
    """
    files: List[ExportFileInfo] = []
    for f in sorted(EXPORTS_DIR.iterdir()):
        if not f.is_file():
            continue
        if extension and not f.suffix.lower() == extension.lower():
            continue
        files.append(_file_info(f))

    # Filter by query number if provided
    if query_number and query_number in QUERY_EXPORT_MAP:
        allowed = set(QUERY_EXPORT_MAP[query_number]["exports"])
        files = [f for f in files if f.filename in allowed]

    # Summary statistics
    total_size = sum(f.size_bytes for f in files)
    ext_counts: Dict[str, int] = {}
    for f in files:
        ext_counts[f.extension] = ext_counts.get(f.extension, 0) + 1

    return {
        "total_files": len(files),
        "total_size": _human_size(total_size),
        "by_extension": ext_counts,
        "files": files,
    }


@app.get("/exports/{filename}", tags=["Exports"])
async def download_export(filename: str):
    """
    Download an export file.

    Returns the file with appropriate Content-Type:
    - ``.geojson`` → ``application/geo+json``
    - ``.png``     → ``image/png``
    - ``.tif``     → ``image/tiff``
    - ``.txt``     → ``text/plain``
    """
    filepath = EXPORTS_DIR / filename
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(404, f"Export file '{filename}' not found")

    # Security: prevent path traversal
    if not filepath.resolve().is_relative_to(EXPORTS_DIR.resolve()):
        raise HTTPException(403, "Access denied")

    ext = filepath.suffix.lower()
    media_types = {
        ".geojson": "application/geo+json",
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".txt": "text/plain",
    }
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        path=str(filepath),
        filename=filename,
        media_type=media_type,
    )


@app.get("/exports/{filename}/preview", tags=["Exports"])
async def preview_export(
    filename: str,
    max_features: int = Query(100, ge=1, le=10000, description="Max GeoJSON features to return"),
):
    """
    Preview an export file without downloading the full thing.

    - **GeoJSON**: Returns parsed JSON (truncated to ``max_features``).
    - **PNG**: Returns base64-encoded image data.
    - **TXT**: Returns the first 5000 characters.
    - **TIFF**: Returns file metadata (size, name) — no inline preview.
    """
    filepath = EXPORTS_DIR / filename
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(404, f"Export file '{filename}' not found")

    if not filepath.resolve().is_relative_to(EXPORTS_DIR.resolve()):
        raise HTTPException(403, "Access denied")

    ext = filepath.suffix.lower()

    if ext == ".geojson":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Truncate features for preview
        if isinstance(data, dict) and "features" in data:
            total = len(data["features"])
            data["features"] = data["features"][:max_features]
            data["_preview"] = {
                "total_features": total,
                "shown_features": min(total, max_features),
                "truncated": total > max_features,
            }
        return JSONResponse(data)

    elif ext == ".png":
        with open(filepath, "rb") as f:
            img_bytes = f.read()
        b64 = base64.b64encode(img_bytes).decode("ascii")
        return {
            "filename": filename,
            "mime_type": "image/png",
            "size_bytes": len(img_bytes),
            "base64": f"data:image/png;base64,{b64}",
        }

    elif ext == ".txt":
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(5000)
        return {
            "filename": filename,
            "mime_type": "text/plain",
            "preview": text,
            "truncated": filepath.stat().st_size > 5000,
        }

    elif ext in (".tif", ".tiff"):
        stat = filepath.stat()
        return {
            "filename": filename,
            "mime_type": "image/tiff",
            "size_bytes": stat.st_size,
            "size_human": _human_size(stat.st_size),
            "note": "TIFF preview not available inline — use /exports/{filename} to download",
        }

    else:
        raise HTTPException(400, f"Preview not supported for extension '{ext}'")


@app.delete("/exports/{filename}", tags=["Exports"])
async def delete_export(filename: str):
    """Delete an export file."""
    filepath = EXPORTS_DIR / filename
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(404, f"Export file '{filename}' not found")
    if not filepath.resolve().is_relative_to(EXPORTS_DIR.resolve()):
        raise HTTPException(403, "Access denied")
    filepath.unlink()
    return {"deleted": filename}


# ============================================================================
# ROUTES — Query-to-Export Mapping
# ============================================================================

@app.get("/analyses", tags=["Analyses"])
async def list_analyses():
    """
    List all 17 analysis types with their expected exports.

    Useful for a frontend to show available analyses and check which
    outputs already exist on disk.
    """
    results = []
    for qnum, info in QUERY_EXPORT_MAP.items():
        existing = [f for f in info["exports"] if (EXPORTS_DIR / f).exists()]
        results.append({
            "query_number": qnum,
            "title": info["title"],
            "description": info["description"],
            "output_types": info["output_types"],
            "expected_exports": info["exports"],
            "existing_exports": existing,
            "complete": len(existing) == len(info["exports"]),
        })
    return {"analyses": results}


@app.get("/analyses/{query_number}", tags=["Analyses"])
async def get_analysis(query_number: int):
    """
    Get details and existing exports for a specific analysis (1–17).
    """
    if query_number not in QUERY_EXPORT_MAP:
        raise HTTPException(404, f"Analysis #{query_number} not found (valid: 1–17)")

    info = QUERY_EXPORT_MAP[query_number]
    export_details = []
    for fname in info["exports"]:
        fpath = EXPORTS_DIR / fname
        if fpath.exists():
            export_details.append(_file_info(fpath))
        else:
            export_details.append({"filename": fname, "exists": False})

    return {
        "query_number": query_number,
        **info,
        "export_details": export_details,
    }


# ============================================================================
# ROUTES — Sessions & Feedback (Langfuse integration)
# ============================================================================

@app.get("/sessions", tags=["Observability"])
async def list_sessions(limit: int = Query(20, ge=1, le=100)):
    """
    List recent Langfuse session IDs from the in-memory query store.

    For full session data, use the Langfuse dashboard directly.
    """
    session_ids = sorted(
        set(e.get("session_id") for e in _query_store.values() if e.get("session_id")),
        reverse=True,
    )[:limit]
    return {"sessions": session_ids}


@app.get("/sessions/{session_id}", tags=["Observability"])
async def get_session(session_id: str):
    """
    Get all queries associated with a Langfuse session ID.
    """
    queries = [
        e for e in _query_store.values()
        if e.get("session_id") == session_id
    ]
    if not queries:
        raise HTTPException(404, f"Session '{session_id}' not found")

    queries = sorted(queries, key=lambda x: x["created_at"])
    return {
        "session_id": session_id,
        "total_queries": len(queries),
        "queries": queries,
    }


@app.post("/feedback", tags=["Observability"])
async def submit_feedback(req: FeedbackRequest):
    """
    Submit user feedback (thumbs up/down) for a completed query.

    This records a score on the Langfuse trace so you can track
    answer quality over time.

    - ``score``: 0.0 (bad) → 1.0 (good)
    - ``comment``: optional free-text
    """
    # Validate that query exists
    if req.query_id:
        entry = _query_store.get(req.query_id)
        if not entry:
            raise HTTPException(404, f"Query {req.query_id} not found")

    # Record in Langfuse
    try:
        from langfuse_observability import score_trace_by_id, lf_client

        trace_id = req.trace_id
        if not trace_id and req.query_id:
            # The trace_id is the same used during the agent run.
            # For a more robust approach, store the trace_id in _query_store
            # when the agent runs. For now, use the score_trace_by_id approach.
            pass

        if trace_id:
            score_trace_by_id(
                trace_id=trace_id,
                name="user_feedback",
                value=req.score,
                data_type="NUMERIC",
                comment=req.comment,
            )
    except Exception as exc:
        # Don't fail the request if Langfuse is down
        print(f"⚠️  Langfuse feedback recording failed: {exc}")

    return {
        "status": "recorded",
        "query_id": req.query_id,
        "score": req.score,
        "comment": req.comment,
    }


# ============================================================================
# ROUTES — GeoJSON-specific utilities
# ============================================================================

@app.get("/geojson", tags=["GeoJSON"])
async def list_geojson_files():
    """List all GeoJSON exports with feature counts."""
    results = []
    for f in sorted(EXPORTS_DIR.glob("*.geojson")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            n_features = len(data.get("features", []))
        except Exception:
            n_features = -1

        info = _file_info(f)
        results.append({
            **info.model_dump(),
            "feature_count": n_features,
        })

    return {"total": len(results), "files": results}


@app.get("/geojson/{filename}/properties", tags=["GeoJSON"])
async def get_geojson_properties(filename: str):
    """
    Return the list of property keys and a sample of values from a GeoJSON file.

    Useful for a frontend to build dynamic filters / column selectors.
    """
    filepath = EXPORTS_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, f"GeoJSON file '{filename}' not found")
    if filepath.suffix.lower() != ".geojson":
        raise HTTPException(400, "Not a GeoJSON file")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    if not features:
        return {"filename": filename, "properties": {}, "sample_count": 0}

    # Collect all property keys and sample values
    all_keys: Dict[str, List[Any]] = {}
    for feat in features[:50]:  # sample first 50
        props = feat.get("properties", {})
        for k, v in props.items():
            all_keys.setdefault(k, []).append(v)

    property_info = {}
    for key, values in all_keys.items():
        non_null = [v for v in values if v is not None]
        property_info[key] = {
            "sample_values": non_null[:5],
            "type": type(non_null[0]).__name__ if non_null else "null",
            "null_count": len(values) - len(non_null),
        }

    return {
        "filename": filename,
        "total_features": len(features),
        "properties": property_info,
        "sample_count": min(len(features), 50),
    }


@app.get("/geojson/{filename}/filter", tags=["GeoJSON"])
async def filter_geojson(
    filename: str,
    property_name: str = Query(..., description="Property key to filter on"),
    value: str = Query(..., description="Value to match (exact, case-insensitive)"),
    max_features: int = Query(500, ge=1, le=10000),
):
    """
    Filter a GeoJSON file by a property value and return matching features.

    Example: ``/geojson/drought_affected_villages_navalgund.geojson/filter?property_name=village_name&value=Annigeri``
    """
    filepath = EXPORTS_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, f"GeoJSON file '{filename}' not found")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    matched = []
    for feat in features:
        prop_val = feat.get("properties", {}).get(property_name)
        if prop_val is not None and str(prop_val).lower() == value.lower():
            matched.append(feat)
            if len(matched) >= max_features:
                break

    return {
        "type": "FeatureCollection",
        "features": matched,
        "_filter": {
            "property": property_name,
            "value": value,
            "total_matched": len(matched),
            "total_features": len(features),
        },
    }


# ============================================================================
# Startup / Shutdown events
# ============================================================================

@app.on_event("startup")
async def on_startup():
    print("🚀 CoreStack API server starting…")
    print(f"📂 Exports directory: {EXPORTS_DIR}")
    print(f"📊 Export files: {len(_list_export_filenames())}")


@app.on_event("shutdown")
async def on_shutdown():
    """Flush Langfuse buffer and shut down the thread pool."""
    try:
        from langfuse_observability import shutdown as lf_shutdown
        lf_shutdown()
        print("✅ Langfuse flushed on shutdown")
    except Exception:
        pass
    _executor.shutdown(wait=False)
    print("✅ API server stopped")


# ============================================================================
# Direct execution (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )

