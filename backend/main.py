from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

try:
    from .history_store import HistoryStore
    from .inference import InferenceEngine
    from .schemas import (
        AnalysisResponse,
        AnalyzeRequest,
        ConversationRequest,
        HistoryEntry,
        ModelInfo,
    )
except ImportError:
    # Supports legacy/cwd-based startup: uvicorn app.main:app from backend/.
    from history_store import HistoryStore
    from inference import InferenceEngine
    from schemas import (
        AnalysisResponse,
        AnalyzeRequest,
        ConversationRequest,
        HistoryEntry,
        ModelInfo,
    )


app = FastAPI(title="EchoSafe API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "ECHOSAFE_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = InferenceEngine()
history = HistoryStore()


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_status": engine.status,
        "mode": engine.mode,
        "database_status": history.status,
    }


@app.get("/api/model-info", response_model=ModelInfo)
def model_info() -> dict:
    return engine.model_info()


@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze(request: AnalyzeRequest) -> dict:
    try:
        result = engine.analyze(request.text, request.context)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if request.save:
        saved = history.add({
            "text": request.text,
            "emotions": result["emotions"],
            "stress_score": result["stress_score"],
            "stress_label": result["stress_label"],
            "uncertainty": result["uncertainty"],
            "explanations": result["explanations"],
            "context": request.context,
            "analysis_type": result["analysis_type"],
            "model_status": result["model_status"],
            "mode": result["mode"],
        })
        result["analysis_id"] = saved["id"]
    return result


@app.post("/api/analyze/conversation", response_model=AnalysisResponse)
def analyze_conversation(request: ConversationRequest) -> dict:
    current = request.messages[-1].text
    context = [message.text for message in request.messages[:-1]][-5:]
    try:
        return engine.analyze(current, context)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/experiments")
def experiments() -> dict:
    path = Path(os.getenv("ECHOSAFE_EXPERIMENTS_FILE", "ml/experiments/experiments.json"))
    if not path.exists():
        return {"experiments": [], "note": "No experiment results have been recorded."}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Experiments file is not valid JSON.")
    return payload if isinstance(payload, dict) else {"experiments": payload}


@app.get("/api/experiments/plots")
def experiment_plots() -> dict:
    path = Path(os.getenv("ECHOSAFE_PLOTS_FILE", "ml/experiments/plots.json"))
    if not path.exists():
        return {"plots": [], "note": "No experiment plots have been generated."}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Plots file is not valid JSON.")
    return payload if isinstance(payload, dict) else {"plots": payload}


@app.get("/api/resources")
def resources() -> dict:
    return {
        "resources": [
            {
                "title": "Talk to someone you trust",
                "description": "A friend, family member, mentor, or colleague can be a meaningful first step.",
                "url": None,
            },
            {
                "title": "Immediate danger",
                "description": "If you may be in immediate danger, contact local emergency services or a trusted person who can stay with you.",
                "url": None,
            },
        ],
        "disclaimer": "EchoSafe is not a clinical service and does not provide diagnosis or crisis triage.",
    }


@app.get("/api/history")
def get_history(limit: int = Query(default=100, ge=1, le=1000)) -> dict:
    return {"entries": history.list()[:limit], "database_status": history.status}


@app.post("/api/history", response_model=HistoryEntry)
def save_history(entry: HistoryEntry) -> dict:
    return history.add(entry.model_dump(exclude={"id", "created_at"}, exclude_none=True))


@app.delete("/api/history/{entry_id}")
def delete_history(entry_id: str) -> dict:
    if not history.delete(entry_id):
        raise HTTPException(status_code=404, detail="History entry not found.")
    return {"deleted": True, "id": entry_id}


@app.delete("/api/history")
def clear_history() -> dict:
    return {"deleted": history.clear()}
