from pydantic import BaseModel
from typing import Optional, Literal


class TrackAnalysis(BaseModel):
    bpm: float
    energy: float
    key: str
    mode: str
    key_confidence: float
    duration: float
    best_outro_time: float
    best_intro_time: float


class TransitionRequest(BaseModel):
    track1_filename: str
    track2_filename: str
    transition_duration: float = 15.0
    overlap_duration: float = 0.5
    style: Literal['smooth', 'drop', 'echo'] = 'smooth'


class TransitionResponse(BaseModel):
    success: bool
    message: str
    output_filename: Optional[str] = None
    duration: Optional[float] = None
    transition_duration: Optional[float] = None
    track1_analysis: Optional[TrackAnalysis] = None
    track2_analysis: Optional[TrackAnalysis] = None
    style: Optional[str] = None
    quality_info: Optional[str] = None
    ai_decisions: Optional[list] = None
    audio_scores: Optional[dict] = None
    mel_model_used: bool = False
    transition_start: Optional[float] = None


class AnalyzeRequest(BaseModel):
    filename: str


class AnalyzeResponse(BaseModel):
    success: bool
    message: str
    analysis: Optional[TrackAnalysis] = None



class HealthResponse(BaseModel):
    status: str
    version: str
    ai_model_loaded: bool = False