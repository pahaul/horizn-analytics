"""
api.py
──────
FastAPI wrapper for the Horizn Analytics brand analysis pipeline.

Exposes three endpoints:

    POST /analyse
        Accepts a URL, scrapes the site (corpus.json is assumed to already
        exist at ./output/corpus.json), runs the full pipeline in the
        background, and returns a run_id immediately.

        In Phase 4 (n8n integration) the scraping step will be triggered
        separately. For now the caller is expected to supply a pre-built
        corpus.json — the URL is passed through to run_pipeline() for
        metadata purposes only.

    GET /profile/{run_id}
        Retrieves the BrandProfile produced by a previous /analyse call.
        run_id is the timestamp string used in the output filename
        (e.g. "20260409T152408").

    GET /health
        Checks whether Ollama is reachable and the default model is loaded.

Design constraints carried forward from the handover:
    - run_pipeline() in pipeline.py is the single entry point.
    - Pydantic models from schemas.py are reused directly as response models.
    - Output files live in ./output/ and are already timestamped by save_profile().
    - The timestamp string is the run_id — the API reads it back from disk.
    - No authentication (local tool; Phase 4 adds n8n).
    - No stop tokens, no tool calling (Gemma 4B constraints).
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

from schemas import BrandProfile

# ── Logging ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────

OUTPUT_DIR     = Path("./output")
OLLAMA_BASE    = "http://localhost:11434"
DEFAULT_MODEL  = "gemma3:4b"
CRITERIA_DIR   = "."


# ── In-memory run registry ────────────────────────────────────────────────
#
# Maps run_id → status string while a job is in progress.
# Once the pipeline finishes, the BrandProfile is on disk; the status entry
# is kept so the client can distinguish "still running" from "never started".

_run_status: dict[str, str] = {}   # run_id → "running" | "done" | "failed"
_run_errors:  dict[str, str] = {}  # run_id → error message (on failure only)


# ── Lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("output directory: %s", OUTPUT_DIR.resolve())
    yield


# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Horizn Analytics API",
    version="0.3.0",
    description=(
        "LLM-powered brand analysis pipeline. "
        "Submit a URL to trigger analysis; poll /profile/{run_id} for results."
    ),
    lifespan=lifespan,
)


# ── Request / response schemas ────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    """Body for POST /analyse."""

    url: HttpUrl = ...
    corpus_path: str = "./output/corpus.json"
    skip_vision: bool = False
    model: str = DEFAULT_MODEL
    criteria_dir: str = CRITERIA_DIR
    ollama_base_url: str = OLLAMA_BASE


class AnalyseResponse(BaseModel):
    """Immediate response from POST /analyse."""

    run_id: str
    status: str   # always "running" at submission time
    message: str


class RunStatusResponse(BaseModel):
    """Response from GET /profile/{run_id} when job is still in progress."""

    run_id: str
    status: str   # "running" | "done" | "failed"
    detail: str   # human-readable summary


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str          # "ok" | "degraded"
    ollama_reachable: bool
    model_available: bool
    model: str
    ollama_url: str


# ── Background task ───────────────────────────────────────────────────────

def _run_pipeline_task(
    run_id: str,
    url: str,
    corpus_path: str,
    skip_vision: bool,
    model: str,
    criteria_dir: str,
    ollama_base_url: str,
) -> None:
    """
    Executes the full pipeline synchronously in a background thread.

    FastAPI's BackgroundTasks runs this in a thread pool, so it does not
    block the event loop. Status is tracked in _run_status / _run_errors.
    The output file path is derived from the run_id that pipeline.py embeds
    in the filename via save_profile().
    """
    from pipeline import run_pipeline  # imported here to avoid circular deps at startup

    _run_status[run_id] = "running"
    logger.info("pipeline task started — run_id=%s  url=%s", run_id, url)

    try:
        output_path = OUTPUT_DIR / f"brand_profile_{run_id}.json"
        run_pipeline(
            corpus_path=corpus_path,
            output_path=str(output_path),
            source_url=url,
            criteria_dir=criteria_dir,
            ollama_base_url=ollama_base_url,
            model=model,
            skip_vision=skip_vision,
        )
        _run_status[run_id] = "done"
        logger.info("pipeline task done — run_id=%s", run_id)

    except Exception as exc:  # noqa: BLE001 — log all failures
        _run_status[run_id] = "failed"
        _run_errors[run_id]  = str(exc)
        logger.error("pipeline task failed — run_id=%s  error=%s", run_id, exc)


# ── Helper: derive run_id from output filename ────────────────────────────

def _run_id_from_output(output_path: Path) -> str:
    """
    Extracts the timestamp portion from a brand_profile filename.

    Example: brand_profile_20260409T152408.json → "20260409T152408"
    Falls back to the full stem if the expected pattern is not found.
    """
    stem = output_path.stem                   # e.g. "brand_profile_20260409T152408"
    parts = stem.split("_", maxsplit=2)       # ["brand", "profile", "20260409T152408"]
    return parts[2] if len(parts) == 3 else stem


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.post(
    "/analyse",
    response_model=AnalyseResponse,
    status_code=202,
    summary="Trigger brand analysis",
    description=(
        "Runs the full pipeline (text + optional vision analysis) in the background. "
        "Returns a run_id immediately. Poll GET /profile/{run_id} for the result."
    ),
)
async def analyse(request: AnalyseRequest, background_tasks: BackgroundTasks):
    from zoneinfo import ZoneInfo
    from datetime import datetime

    ts = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y%m%dT%H%M%S")
    run_id = ts

    _run_status[run_id] = "queued"

    background_tasks.add_task(
        _run_pipeline_task,
        run_id       = run_id,
        url          = str(request.url),
        corpus_path  = request.corpus_path,
        skip_vision  = request.skip_vision,
        model        = request.model,
        criteria_dir = request.criteria_dir,
        ollama_base_url = request.ollama_base_url,
    )

    logger.info("analysis queued — run_id=%s", run_id)
    return AnalyseResponse(
        run_id  = run_id,
        status  = "queued",
        message = f"Pipeline started. Poll GET /profile/{run_id} for the result.",
    )


@app.get(
    "/profile/{run_id}",
    summary="Retrieve a brand profile",
    description=(
        "Returns the BrandProfile for a completed run, or a status object "
        "if the run is still in progress or has failed."
    ),
)
async def get_profile(run_id: str):
    # -- Check for a completed output file first (covers runs from previous
    #    server sessions that are no longer in _run_status). -----------------
    profile_path = OUTPUT_DIR / f"brand_profile_{run_id}.json"

    if profile_path.exists():
        with open(profile_path, encoding="utf-8") as f:
            data = json.load(f)
        # Validate through Pydantic to guarantee schema compliance.
        profile = BrandProfile(**data)
        return JSONResponse(content=profile.model_dump())

    # -- Not on disk yet — check in-memory status. --------------------------
    status = _run_status.get(run_id)

    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"No run found with run_id='{run_id}'. "
                   "Check the ID or start a new run via POST /analyse.",
        )

    if status in ("running", "queued"):
        return JSONResponse(
            status_code=202,
            content=RunStatusResponse(
                run_id = run_id,
                status = status,
                detail = "Pipeline is still running. Try again in a moment.",
            ).model_dump(),
        )

    # status == "failed"
    error_detail = _run_errors.get(run_id, "Unknown error")
    raise HTTPException(
        status_code=500,
        detail=f"Pipeline failed for run_id='{run_id}': {error_detail}",
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Checks whether Ollama is reachable and the default model is loaded.",
)
async def health(
    ollama_base_url: str = OLLAMA_BASE,
    model: str = DEFAULT_MODEL,
):
    ollama_reachable = False
    model_available  = False

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # /api/tags lists locally available models
            resp = await client.get(f"{ollama_base_url}/api/tags")
            if resp.status_code == 200:
                ollama_reachable = True
                tags = resp.json().get("models", [])
                model_available = any(
                    m.get("name", "").startswith(model) for m in tags
                )
    except (httpx.ConnectError, httpx.TimeoutException):
        pass   # both flags remain False

    overall = "ok" if (ollama_reachable and model_available) else "degraded"

    return HealthResponse(
        status           = overall,
        ollama_reachable = ollama_reachable,
        model_available  = model_available,
        model            = model,
        ollama_url       = ollama_base_url,
    )


# ── Optional: list recent runs ────────────────────────────────────────────

@app.get(
    "/runs",
    summary="List recent runs",
    description="Returns run_ids and statuses for all runs known to this server session, "
                "plus any brand_profile_*.json files already on disk.",
)
async def list_runs():
    # Discover runs from disk that may predate this server session.
    disk_runs = {
        _run_id_from_output(p): "done"
        for p in sorted(OUTPUT_DIR.glob("brand_profile_*.json"))
    }
    # Merge with in-memory registry (in-memory wins on conflict).
    merged = {**disk_runs, **_run_status}
    return {"runs": [{"run_id": k, "status": v} for k, v in sorted(merged.items())]}


# ── Dev entrypoint ────────────────────────────────────────────────────────
#
# Run with:  python api.py
# Or:        uvicorn api:app --reload

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
