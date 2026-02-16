"""Module de chargement des fichiers audio."""

import os
import librosa
import numpy as np
from src.utils.config import SAMPLE_RATE, SUPPORTED_FORMATS


class AudioLoader:
    """Classe pour charger et valider les fichiers audio."""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        """
        Initialise le loader audio.
        
        Args:
            sample_rate: Fréquence d'échantillonnage cible (Hz)
        """
        self.sample_rate = sample_rate
    
    def load(self, file_path: str) -> tuple[np.ndarray, int]:
        """
        Charge un fichier audio.
        
        Args:
            file_path: Chemin vers le fichier audio
            
        Returns:
            tuple: (signal audio, sample rate)
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si le format n'est pas supporté
        """
        # Vérifier que le fichier existe
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
        
        # Vérifier le format
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(f"Format non supporté: {ext}. Formats acceptés: {SUPPORTED_FORMATS}")
        
        # Charger l'audio avec librosa
        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        
        print(f"✓ Audio chargé: {os.path.basename(file_path)}")
        print(f"  - Durée: {len(audio) / sr:.2f} secondes")
        print(f"  - Sample rate: {sr} Hz")
        
        return audio, sr
    
    def load_segment(self, file_path: str, start: float, duration: float) -> tuple[np.ndarray, int]:
        """
        Charge un segment spécifique d'un fichier audio.
        
        Args:
            file_path: Chemin vers le fichier audio
            start: Position de départ en secondes
            duration: Durée à charger en secondes
            
        Returns:
            tuple: (signal audio, sample rate)
        """
        audio, sr = librosa.load(
            file_path, 
            sr=self.sample_rate, 
            mono=True,
            offset=start,
            duration=duration
        )
        
        return audio, sr
    
    def get_duration(self, file_path: str) -> float:
        """
        Retourne la durée d'un fichier audio sans le charger entièrement.
        
        Args:
            file_path: Chemin vers le fichier audio
            
        Returns:
            float: Durée en secondes
        """
        return librosa.get_duration(path=file_path)


# Fonction utilitaire pour un usage simple
def load_audio(file_path: str, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """
    Fonction simple pour charger un fichier audio.
    
    Args:
        file_path: Chemin vers le fichier audio
        sample_rate: Fréquence d'échantillonnage cible
        
    Returns:
        tuple: (signal audio, sample rate)
    """
    loader = AudioLoader(sample_rate)
    return loader.load(file_path)