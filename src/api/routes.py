"""Routes de l'API."""

import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from src.api.schemas import (
    TransitionRequest, TransitionResponse,
    AnalyzeRequest, AnalyzeResponse,
    TrackAnalysis, HealthResponse
)
from src.audio.loader import AudioLoader
from src.analysis.feature_extractor import FeatureExtractor
from src.analysis.beat_detector import BeatDetector
from src.analysis.key_analyzer import KeyAnalyzer
from src.transition.generator import TransitionGenerator
from src.utils.config import SAMPLE_RATE

# Créer le router
router = APIRouter()

# Dossiers de travail
UPLOAD_DIR = "data/input"
OUTPUT_DIR = "data/output"

# Initialiser les modules
loader = AudioLoader(SAMPLE_RATE)
feature_extractor = FeatureExtractor(SAMPLE_RATE)
beat_detector = BeatDetector(SAMPLE_RATE)
key_analyzer = KeyAnalyzer(SAMPLE_RATE)
generator = TransitionGenerator(SAMPLE_RATE)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérifie que l'API fonctionne."""
    # Vérifier si le modèle IA est chargé
    ai_loaded = generator.ai_generator.vae_model is not None
    
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        ai_model_loaded=ai_loaded
    )


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload un fichier audio."""
    allowed_extensions = ['.mp3', '.wav', '.flac', '.ogg']
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté. Formats acceptés: {allowed_extensions}"
        )
    
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    return {
        "success": True,
        "filename": unique_filename,
        "original_name": file.filename,
        "size": len(content)
    }


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_track(request: AnalyzeRequest):
    """Analyse un morceau audio."""
    file_path = os.path.join(UPLOAD_DIR, request.filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Fichier non trouvé: {request.filename}"
        )
    
    try:
        audio, sr = loader.load(file_path)
        
        features = feature_extractor.extract_all(audio)
        key_info = key_analyzer.detect_key(audio)
        outro_point = beat_detector.get_best_outro_point(audio)
        intro_point = beat_detector.get_best_intro_point(audio)
        
        analysis = TrackAnalysis(
            bpm=features['bpm'],
            energy=features['energy'],
            key=key_info['key'],
            mode=key_info['mode'],
            key_confidence=key_info['confidence'],
            duration=features['duration'],
            best_outro_time=outro_point['time'],
            best_intro_time=intro_point['time']
        )
        
        return AnalyzeResponse(
            success=True,
            message="Analyse terminée",
            analysis=analysis
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'analyse: {str(e)}"
        )


@router.post("/generate", response_model=TransitionResponse)
async def generate_transition(request: TransitionRequest):
    """Génère une transition entre deux morceaux."""
    track1_path = os.path.join(UPLOAD_DIR, request.track1_filename)
    track2_path = os.path.join(UPLOAD_DIR, request.track2_filename)
    
    if not os.path.exists(track1_path):
        raise HTTPException(status_code=404, detail=f"Track 1 non trouvé")
    
    if not os.path.exists(track2_path):
        raise HTTPException(status_code=404, detail=f"Track 2 non trouvé")
    
    try:
        output_filename = f"transition_{uuid.uuid4().hex}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Générer la transition
        result = generator.generate_transition(
            track1_path,
            track2_path,
            transition_duration=request.transition_duration,
            output_path=output_path,
            style=request.style,
            overlap_duration=request.overlap_duration
        )
        
        # Construire la réponse
        track1_analysis = TrackAnalysis(
            bpm=result['analysis_track1']['bpm'],
            energy=result['analysis_track1']['energy'],
            key=result['analysis_track1']['key'],
            mode=result['analysis_track1']['mode'],
            key_confidence=result['analysis_track1']['key_confidence'],
            duration=result['analysis_track1']['duration'],
            best_outro_time=result['analysis_track1']['best_outro']['time'],
            best_intro_time=result['analysis_track1']['best_intro']['time']
        )
        
        track2_analysis = TrackAnalysis(
            bpm=result['analysis_track2']['bpm'],
            energy=result['analysis_track2']['energy'],
            key=result['analysis_track2']['key'],
            mode=result['analysis_track2']['mode'],
            key_confidence=result['analysis_track2']['key_confidence'],
            duration=result['analysis_track2']['duration'],
            best_outro_time=result['analysis_track2']['best_outro']['time'],
            best_intro_time=result['analysis_track2']['best_intro']['time']
        )
        
        # Info sur le modèle IA
        ai_info = "Modèle VAE" if generator.ai_generator.vae_model else "Mode fallback (effets)"
        
        return TransitionResponse(
            success=True,
            message="Transition générée avec succès",
            output_filename=output_filename,
            duration=result['duration'],
            transition_duration=result['transition_duration'],
            track1_analysis=track1_analysis,
            track2_analysis=track2_analysis,
            style=result['style'],
            quality_info=ai_info
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération: {str(e)}"
        )


@router.get("/download/{filename}")
async def download_file(filename: str):
    """Télécharge un fichier généré."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier non trouvé")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=filename
    )


@router.get("/files")
async def list_files():
    """Liste tous les fichiers disponibles."""
    input_files = os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else []
    output_files = os.listdir(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else []
    
    return {
        "input_files": input_files,
        "output_files": output_files
    }


@router.get("/styles")
async def get_styles():
    """Retourne les styles de transition disponibles."""
    return {
        "styles": [
            {
                "id": "smooth",
                "name": "Smooth",
                "description": "Transition douce et progressive avec filtres"
            },
            {
                "id": "drop",
                "name": "Drop",
                "description": "Build-up puis drop style EDM"
            },
            {
                "id": "echo",
                "name": "Echo",
                "description": "Dissolution dans l'écho et réverb"
            }
        ]
    }


@router.get("/model/status")
async def model_status():
    """Retourne le statut du modèle IA."""
    vae_loaded = generator.ai_generator.vae_model is not None
    
    return {
        "vae_model_loaded": vae_loaded,
        "model_path": generator.ai_generator.model_path,
        "device": generator.ai_generator.device,
        "mode": "VAE" if vae_loaded else "Fallback (effets audio)"
    }