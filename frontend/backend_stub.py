"""
Smart Retail Intelligence Network - backend stub (FastAPI)
=========================================================
Serves the exact API the frontend expects, using mock_data.json.
Replace each handler's body with calls to your real models / database; keep the response shapes.

Run:
    pip install fastapi uvicorn
    uvicorn backend_stub:app --reload --port 8000
Then open http://localhost:8000  (serves SmartRetail_Frontend.html)
or open the HTML file directly and set Settings > Live API > http://localhost:8000
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

HERE = Path(__file__).parent
DATA: dict[str, Any] = json.loads((HERE / "mock_data.json").read_text(encoding="utf-8"))
TASKS: list[dict[str, Any]] = DATA["tasks"]  # in-memory; swap for a DB table

app = FastAPI(title="Smart Retail Intelligence Network API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])  # tighten in production


class TaskCreate(BaseModel):
    type: str
    sku_id: str
    params: dict[str, Any] = {}
    impact_rs: float = 0
    urgency: int = 3
    minutes: int = 8
    role: str = "floor"


class TaskUpdate(BaseModel):
    status: Literal["open", "done", "not_possible", "snoozed"]
    reason: Optional[str] = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "SmartRetail_Frontend.html")


@app.get("/api/ping")
def ping() -> dict[str, Any]:
    return {"ok": True, "version": app.version}


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    # TODO: compute from forecasts / batches / tasks tables
    return {**DATA["summary"], "open_tasks": sum(t["status"] == "open" for t in TASKS), "sample_data": True}


@app.get("/api/skus")
def skus() -> list[dict[str, Any]]:
    # TODO: product master joined with the SKU health classifier output
    return DATA["skus"]


@app.get("/api/skus/{sku_id}/series")
def series(sku_id: str) -> dict[str, Any]:
    # TODO: last 90 days of actuals + P10/P50/P90 from the forecaster (LightGBM Tweedie + quantile models)
    if sku_id not in DATA["series"]:
        raise HTTPException(404, "Unknown SKU")
    return DATA["series"][sku_id]


@app.get("/api/batches")
def batches() -> list[dict[str, Any]]:
    # TODO: inventory batches + expiry risk estimator (FEFO, expected unsold units)
    return DATA["batches"]


@app.get("/api/spikes")
def spikes() -> list[dict[str, Any]]:
    # TODO: robust z-score / changepoint detector output
    return DATA["spikes"]


@app.get("/api/suppliers")
def suppliers() -> list[dict[str, Any]]:
    # TODO: OTIF, fill rate and lead-time stats from purchase orders and receipts
    return DATA["suppliers"]


@app.get("/api/tasks")
def list_tasks() -> list[dict[str, Any]]:
    return TASKS


@app.post("/api/tasks")
def create_task(body: TaskCreate) -> dict[str, Any]:
    task = {"id": f"t{int(time.time() * 1000)}", "status": "open", **body.model_dump()}
    task["priority"] = round(task["impact_rs"] * task["urgency"] / max(task["minutes"], 1))
    TASKS.insert(0, task)
    return task


@app.patch("/api/tasks/{task_id}")
def update_task(task_id: str, body: TaskUpdate) -> dict[str, Any]:
    for task in TASKS:
        if task["id"] == task_id:
            task["status"] = body.status
            if body.reason:
                task["reason"] = body.reason  # store as a training label for the action engine
            elif body.status == "open":
                task.pop("reason", None)
            return task
    raise HTTPException(404, "Unknown task")


@app.get("/api/map")
def neural_map() -> dict[str, Any]:
    return DATA["map"]
