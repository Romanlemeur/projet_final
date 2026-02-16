"""Module de crossfade - fondu enchaîné intelligent."""

import numpy as np
from src.utils.config import SAMPLE_RATE, DEFAULT_TRANSITION_DURATION


class CrossFader:
    """Classe pour créer des fondus enchaînés entre deux morceaux."""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
    
    def create_fade_curve(self, length: int, curve_type: str = 'equal_power') -> np.ndarray:
        """Crée une courbe de fade."""
        if length <= 0:
            return np.array([])
            
        t = np.linspace(0, 1, length)
        
        if curve_type == 'linear':
            return t
        elif curve_type == 'equal_power':
            return np.sin(t * np.pi / 2)
        elif curve_type == 's_curve':
            return (1 - np.cos(t * np.pi)) / 2
        elif curve_type == 'exponential':
            return t ** 2
        else:
            return t
    
    def simple_crossfade(self, audio1: np.ndarray, audio2: np.ndarray,
                         fade_duration: float = 8.0,
                         curve_type: str = 'equal_power') -> dict:
        """
        Crée un crossfade simple entre la FIN de audio1 et le DÉBUT de audio2.
        
        Args:
            audio1: Premier morceau complet
            audio2: Deuxième morceau complet
            fade_duration: Durée du crossfade en secondes
            curve_type: Type de courbe
            
        Returns:
            dict: Audio de transition et métadonnées
        """
        fade_samples = int(fade_duration * self.sample_rate)
        
        # S'assurer qu'on ne dépasse pas la taille des morceaux
        max_fade = min(len(audio1) // 2, len(audio2) // 2, fade_samples)
        fade_samples = max(max_fade, int(2.0 * self.sample_rate))  # Minimum 2 secondes
        
        print(f"  📐 Fade samples: {fade_samples} ({fade_samples / self.sample_rate:.1f}s)")
        
        # Extraire les segments
        outro = audio1[-fade_samples:]  # Fin du morceau 1
        intro = audio2[:fade_samples]   # Début du morceau 2
        
        # Créer les courbes
        fade_out_curve = 1 - self.create_fade_curve(fade_samples, curve_type)
        fade_in_curve = self.create_fade_curve(fade_samples, curve_type)
        
        # Mixer
        transition = (outro * fade_out_curve) + (intro * fade_in_curve)
        
        print(f"  🔀 Crossfade créé: {fade_samples / self.sample_rate:.1f}s ({fade_samples} samples)")
        
        return {
            'audio': transition,
            'duration': fade_samples / self.sample_rate,
            'fade_samples': fade_samples,
            'curve_type': curve_type
        }
    
    def apply_fade_in(self, audio: np.ndarray, duration: float = 2.0,
                      curve_type: str = 'equal_power') -> np.ndarray:
        """Applique un fade in au début."""
        fade_length = min(int(duration * self.sample_rate), len(audio))
        result = audio.copy()
        result[:fade_length] *= self.create_fade_curve(fade_length, curve_type)
        return result
    
    def apply_fade_out(self, audio: np.ndarray, duration: float = 2.0,
                       curve_type: str = 'equal_power') -> np.ndarray:
        """Applique un fade out à la fin."""
        fade_length = min(int(duration * self.sample_rate), len(audio))
        result = audio.copy()
        result[-fade_length:] *= (1 - self.create_fade_curve(fade_length, curve_type))
        return result