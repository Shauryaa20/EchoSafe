from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    context: List[str] = Field(default_factory=list, max_length=20)
    save: bool = False


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = "user"
    text: str = Field(..., min_length=1, max_length=5000)


class ConversationRequest(BaseModel):
    messages: List[ConversationMessage] = Field(..., min_length=1, max_length=50)


class EmotionScore(BaseModel):
    label: str
    score: float


class Uncertainty(BaseModel):
    confidence: float
    is_uncertain: bool
    method: str


class Explanation(BaseModel):
    text: str
    importance: float


class AnalysisResponse(BaseModel):
    emotions: List[EmotionScore]
    stress_score: float
    stress_label: str
    uncertainty: Uncertainty
    explanations: List[Explanation]
    model_status: Literal["available", "unavailable"]
    mode: Literal["checkpoint", "unavailable"]
    context: List[str] = Field(default_factory=list)
    analysis_type: Literal["individual", "contextual", "journal"] = "individual"
    analysis_id: Optional[str] = None
    disclaimer: str


class HistoryEntry(BaseModel):
    id: Optional[str] = None
    text: str = Field(..., min_length=1, max_length=5000)
    emotions: Optional[List[EmotionScore]] = None
    stress_score: Optional[float] = None
    stress_label: Optional[str] = None
    uncertainty: Optional[Uncertainty] = None
    explanations: Optional[List[Explanation]] = None
    context: List[str] = Field(default_factory=list)
    analysis_type: Literal["individual", "contextual", "journal"] = "individual"
    model_status: Optional[str] = None
    mode: Optional[str] = None
    created_at: Optional[datetime] = None


class ModelInfo(BaseModel):
    status: Literal["available", "unavailable"]
    mode: Literal["checkpoint", "demo_fallback"]
    checkpoint: Optional[str] = None
    architecture: Dict[str, object]
