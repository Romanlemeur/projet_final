"""Module de détection des beats et points de transition."""

import numpy as np
import librosa
from src.utils.config import SAMPLE_RATE, HOP_LENGTH


class BeatDetector:
    """Classe pour détecter les beats et trouver les points de transition."""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        """
        Initialise le détecteur de beats.
        
        Args:
            sample_rate: Fréquence d'échantillonnage (Hz)
        """
        self.sample_rate = sample_rate
        self.hop_length = HOP_LENGTH
    
    def detect_beats(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Détecte les beats dans le signal audio.
        
        Args:
            audio: Signal audio
            
        Returns:
            tuple: (temps des beats en secondes, indices des frames)
        """
        tempo, beat_frames = librosa.beat.beat_track(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Convertir les frames en temps (secondes)
        beat_times = librosa.frames_to_time(
            beat_frames,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        return beat_times, beat_frames
    
    def find_downbeats(self, audio: np.ndarray) -> np.ndarray:
        """
        Trouve les downbeats (premiers temps des mesures).
        Idéal pour les points de transition.
        
        Args:
            audio: Signal audio
            
        Returns:
            np.ndarray: Temps des downbeats en secondes
        """
        beat_times, _ = self.detect_beats(audio)
        
        # Estimation simple : un downbeat tous les 4 beats (mesure 4/4)
        downbeats = beat_times[::4]
        
        return downbeats
    
    def find_transition_points(self, audio: np.ndarray, 
                                min_time: float = 30.0,
                                max_time: float = None) -> list[dict]:
        """
        Trouve les meilleurs points pour commencer/finir une transition.
        
        Args:
            audio: Signal audio
            min_time: Temps minimum en secondes (éviter le début)
            max_time: Temps maximum en secondes (éviter la fin)
            
        Returns:
            list: Liste de points de transition potentiels
        """
        duration = len(audio) / self.sample_rate
        
        if max_time is None:
            max_time = duration - 30.0  # Éviter les 30 dernières secondes
        
        # Obtenir les downbeats
        downbeats = self.find_downbeats(audio)
        
        # Calculer l'énergie à chaque point
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        
        transition_points = []
        
        for beat_time in downbeats:
            if min_time <= beat_time <= max_time:
                # Trouver l'index correspondant dans la courbe RMS
                frame_idx = int(beat_time * self.sample_rate / self.hop_length)
                frame_idx = min(frame_idx, len(rms) - 1)
                
                energy_at_point = float(rms[frame_idx])
                
                transition_points.append({
                    'time': round(beat_time, 3),
                    'energy': round(energy_at_point, 4),
                    'type': 'downbeat'
                })
        
        return transition_points
    
    def get_best_outro_point(self, audio: np.ndarray) -> dict:
        """
        Trouve le meilleur point pour commencer la sortie (outro).
        Cherche un point avec une énergie décroissante.
        
        Args:
            audio: Signal audio
            
        Returns:
            dict: Meilleur point de sortie
        """
        duration = len(audio) / self.sample_rate
        
        # Chercher dans le dernier tiers du morceau
        min_time = duration * 0.6
        max_time = duration - 10.0
        
        points = self.find_transition_points(audio, min_time, max_time)
        
        if not points:
            return {'time': duration - 15.0, 'energy': 0.5, 'type': 'fallback'}
        
        # Retourner le point avec l'énergie la plus basse
        return min(points, key=lambda x: x['energy'])
    
    def get_best_intro_point(self, audio: np.ndarray) -> dict:
        """
        Trouve le meilleur point pour la transition d'entrée (intro).
        Cherche un point après l'intro avec une montée d'énergie.
        
        Args:
            audio: Signal audio
            
        Returns:
            dict: Meilleur point d'entrée
        """
        duration = len(audio) / self.sample_rate
        
        # Chercher dans le premier tiers du morceau
        min_time = 10.0
        max_time = duration * 0.4
        
        points = self.find_transition_points(audio, min_time, max_time)
        
        if not points:
            return {'time': 15.0, 'energy': 0.5, 'type': 'fallback'}
        
        # Retourner le point avec l'énergie la plus haute
        return max(points, key=lambda x: x['energy'])


def detect_beats(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Fonction simple pour détecter les beats.
    
    Args:
        audio: Signal audio
        sample_rate: Fréquence d'échantillonnage
        
    Returns:
        np.ndarray: Temps des beats en secondes
    """
    detector = BeatDetector(sample_rate)
    beat_times, _ = detector.detect_beats(audio)
    return beat_times