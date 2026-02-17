"""Schémas Pydantic pour l'API."""

from pydantic import BaseModel
from typing import Optional, Literal


class TrackAnalysis(BaseModel):
    """Résultat de l'analyse d'un morceau."""
    bpm: float
    energy: float
    key: str
    mode: str
    key_confidence: float
    duration: float
    best_outro_time: float
    best_intro_time: float


class TransitionRequest(BaseModel):
    """Requête de génération de transition."""
    track1_filename: str
    track2_filename: str
    transition_duration: float = 15.0
    overlap_duration: float = 3.0
    style: Literal['smooth', 'drop', 'echo'] = 'smooth'


class TransitionResponse(BaseModel):
    """Réponse après génération de transition."""
    success: bool
    message: str
    output_filename: Optional[str] = None
    duration: Optional[float] = None
    transition_duration: Optional[float] = None
    track1_analysis: Optional[TrackAnalysis] = None
    track2_analysis: Optional[TrackAnalysis] = None
    style: Optional[str] = None
    quality_info: Optional[str] = None


class AnalyzeRequest(BaseModel):
    """Requête d'analyse d'un morceau."""
    filename: str


class AnalyzeResponse(BaseModel):
    """Réponse d'analyse."""
    success: bool
    message: str
    analysis: Optional[TrackAnalysis] = None


class HealthResponse(BaseModel):
    """Réponse du health check."""
    status: str
    version: str
    ai_model_loaded: bool = False