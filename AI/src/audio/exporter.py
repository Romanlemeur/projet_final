"""Module d'export des fichiers audio."""

import os
import numpy as np
import soundfile as sf
from src.utils.config import SAMPLE_RATE


class AudioExporter:
    """Classe pour exporter les fichiers audio."""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):

        self.sample_rate = sample_rate
    
    def export_wav(self, audio: np.ndarray, output_path: str) -> str:
       
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        audio_normalized = self._normalize(audio)
        
        sf.write(output_path, audio_normalized, self.sample_rate)
        
        print(f"Audio exporté: {output_path}")
        print(f"Durée: {len(audio) / self.sample_rate:.2f} secondes")
        
        return output_path
    
    def _normalize(self, audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        peak = np.max(np.abs(audio))
        if peak > 0:
            return audio * (target_peak / peak)
        return audio


def export_audio(audio: np.ndarray, output_path: str, sample_rate: int = SAMPLE_RATE) -> str:
    exporter = AudioExporter(sample_rate)
    return exporter.export_wav(audio, output_path)