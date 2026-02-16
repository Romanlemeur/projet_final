"""Module d'extraction des caractéristiques audio."""

import numpy as np
import librosa
from src.utils.config import SAMPLE_RATE, HOP_LENGTH, N_FFT


class FeatureExtractor:
    """Classe pour extraire les caractéristiques d'un signal audio."""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        """
        Initialise l'extracteur de features.
        
        Args:
            sample_rate: Fréquence d'échantillonnage (Hz)
        """
        self.sample_rate = sample_rate
        self.hop_length = HOP_LENGTH
        self.n_fft = N_FFT
    
    def extract_all(self, audio: np.ndarray) -> dict:
        """
        Extrait toutes les caractéristiques d'un signal audio.
        
        Args:
            audio: Signal audio (numpy array)
            
        Returns:
            dict: Dictionnaire contenant toutes les features
        """
        features = {
            'bpm': self.extract_bpm(audio),
            'energy': self.extract_energy(audio),
            'rms_mean': self.extract_rms(audio),
            'duration': len(audio) / self.sample_rate
        }
        
        return features
    
    def extract_bpm(self, audio: np.ndarray) -> float:
        """
        Extrait le BPM (tempo) du signal audio.
        
        Args:
            audio: Signal audio
            
        Returns:
            float: BPM estimé
        """
        tempo, _ = librosa.beat.beat_track(
            y=audio, 
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # librosa peut retourner un array, on prend la première valeur
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])
        
        return round(tempo, 2)
    
    def extract_energy(self, audio: np.ndarray) -> float:
        """
        Calcule l'énergie moyenne du signal.
        
        Args:
            audio: Signal audio
            
        Returns:
            float: Énergie moyenne (0.0 à 1.0)
        """
        # Calcul de l'énergie RMS
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        
        # Normaliser entre 0 et 1
        energy = float(np.mean(rms))
        
        # Mapper vers une échelle plus intuitive (0-1)
        # Les valeurs RMS sont généralement entre 0 et 0.3
        energy_normalized = min(energy / 0.2, 1.0)
        
        return round(energy_normalized, 3)
    
    def extract_rms(self, audio: np.ndarray) -> float:
        """
        Calcule la valeur RMS moyenne (volume perçu).
        
        Args:
            audio: Signal audio
            
        Returns:
            float: Valeur RMS moyenne
        """
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        return round(float(np.mean(rms)), 4)
    
    def extract_energy_curve(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrait la courbe d'énergie au cours du temps.
        Utile pour trouver les meilleurs points de transition.
        
        Args:
            audio: Signal audio
            
        Returns:
            np.ndarray: Courbe d'énergie
        """
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        return rms


def extract_features(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    """
    Fonction simple pour extraire les features d'un audio.
    
    Args:
        audio: Signal audio
        sample_rate: Fréquence d'échantillonnage
        
    Returns:
        dict: Caractéristiques extraites
    """
    extractor = FeatureExtractor(sample_rate)
    return extractor.extract_all(audio)