"""Module de beatmatching - synchronisation des tempos."""

import numpy as np
import librosa
from src.utils.config import SAMPLE_RATE


class BeatMatcher:
    """Classe pour synchroniser le tempo entre deux morceaux."""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
       
        self.sample_rate = sample_rate
    
    def time_stretch(self, audio: np.ndarray, source_bpm: float, target_bpm: float) -> np.ndarray:
        """
        Ajuste le tempo d'un audio sans changer la hauteur.
        
        """
        rate = target_bpm / source_bpm
        
        rate = np.clip(rate, 0.8, 1.2)
        
        if abs(rate - 1.0) < 0.01:
            return audio
        
        audio_stretched = librosa.effects.time_stretch(audio, rate=rate)
        
        print(f"Time stretch: {source_bpm:.1f} → {target_bpm:.1f} BPM (ratio: {rate:.3f})")
        
        return audio_stretched
    
    def calculate_transition_bpm(self, bpm1: float, bpm2: float, 
                                  method: str = 'average') -> float:
        
        if method == 'average':
            return (bpm1 + bpm2) / 2
        elif method == 'first':
            return bpm1
        elif method == 'second':
            return bpm2
        elif method == 'gradual':
            return (bpm1, bpm2)
        else:
            return (bpm1 + bpm2) / 2
    
    def align_to_beat(self, audio: np.ndarray, beat_times: np.ndarray, 
                      target_time: float) -> int:
       
        closest_beat_idx = np.argmin(np.abs(beat_times - target_time))
        closest_beat_time = beat_times[closest_beat_idx]
        
        sample_idx = int(closest_beat_time * self.sample_rate)
        
        return sample_idx
    
    def prepare_tracks_for_transition(self, audio1: np.ndarray, bpm1: float,
                                       audio2: np.ndarray, bpm2: float,
                                       method: str = 'average') -> tuple:
        
        target_bpm = self.calculate_transition_bpm(bpm1, bpm2, method)
        
        print(f"Synchronisation des tempos:")
        print(f"Track 1: {bpm1:.1f} BPM")
        print(f"Track 2: {bpm2:.1f} BPM")
        print(f"Cible: {target_bpm:.1f} BPM")
        
        audio1_adjusted = self.time_stretch(audio1, bpm1, target_bpm)
        audio2_adjusted = self.time_stretch(audio2, bpm2, target_bpm)
        
        return audio1_adjusted, audio2_adjusted, target_bpm