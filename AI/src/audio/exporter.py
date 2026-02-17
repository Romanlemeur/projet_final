"""Module d'export des fichiers audio."""

import os
import numpy as np
import soundfile as sf
from src.utils.config import SAMPLE_RATE


class AudioExporter:
    """Classe pour exporter les fichiers audio."""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        """
        Initialise l'exporteur audio.
        
        Args:
            sample_rate: Fréquence d'échantillonnage (Hz)
        """
        self.sample_rate = sample_rate
    
    def export_wav(self, audio: np.ndarray, output_path: str) -> str:
        """
        Exporte un signal audio en fichier WAV.
        
        Args:
            audio: Signal audio (numpy array)
            output_path: Chemin de sortie
            
        Returns:
            str: Chemin du fichier créé
        """
        # Créer le dossier si nécessaire
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Normaliser l'audio pour éviter la saturation
        audio_normalized = self._normalize(audio)
        
        # Exporter
        sf.write(output_path, audio_normalized, self.sample_rate)
        
        print(f"✓ Audio exporté: {output_path}")
        print(f"  - Durée: {len(audio) / self.sample_rate:.2f} secondes")
        
        return output_path
    
    def _normalize(self, audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """
        Normalise l'audio pour éviter la saturation.
        
        Args:
            audio: Signal audio
            target_peak: Niveau de crête cible (0.0 à 1.0)
            
        Returns:
            np.ndarray: Audio normalisé
        """
        peak = np.max(np.abs(audio))
        if peak > 0:
            return audio * (target_peak / peak)
        return audio


# Fonction utilitaire pour un usage simple
def export_audio(audio: np.ndarray, output_path: str, sample_rate: int = SAMPLE_RATE) -> str:
    """
    Fonction simple pour exporter un fichier audio.
    
    Args:
        audio: Signal audio (numpy array)
        output_path: Chemin de sortie
        sample_rate: Fréquence d'échantillonnage
        
    Returns:
        str: Chemin du fichier créé
    """
    exporter = AudioExporter(sample_rate)
    return exporter.export_wav(audio, output_path)