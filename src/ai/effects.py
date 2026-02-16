"""Effets audio pour les transitions."""

import numpy as np
from scipy.signal import butter, filtfilt, lfilter
from src.utils.config import SAMPLE_RATE


class AudioEffects:
    """Effets audio pour améliorer les transitions."""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
    
    def apply_filter_sweep(self, audio: np.ndarray, 
                           start_freq: int, end_freq: int,
                           filter_type: str = 'low') -> np.ndarray:
        """
        Applique un sweep de filtre progressif.
        
        Args:
            audio: Signal audio
            start_freq: Fréquence de départ (Hz)
            end_freq: Fréquence de fin (Hz)
            filter_type: 'low' ou 'high'
            
        Returns:
            np.ndarray: Audio filtré
        """
        n_samples = len(audio)
        output = np.zeros(n_samples)
        
        # Diviser en segments et appliquer des filtres progressifs
        n_segments = 50
        segment_size = n_samples // n_segments
        
        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, n_samples)
            
            # Interpoler la fréquence de coupure
            t = i / (n_segments - 1)
            current_freq = start_freq + (end_freq - start_freq) * t
            current_freq = max(20, min(current_freq, self.sample_rate // 2 - 100))
            
            # Appliquer le filtre
            nyquist = self.sample_rate / 2
            normalized_freq = current_freq / nyquist
            normalized_freq = max(0.01, min(normalized_freq, 0.99))
            
            b, a = butter(2, normalized_freq, btype=filter_type)
            
            segment = audio[start_idx:end_idx]
            if len(segment) > 0:
                # Padding pour éviter les artefacts
                pad_len = min(100, len(segment) // 2)
                if pad_len > 0 and len(segment) > pad_len * 2:
                    padded = np.pad(segment, pad_len, mode='edge')
                    filtered = filtfilt(b, a, padded)
                    output[start_idx:end_idx] = filtered[pad_len:-pad_len]
                else:
                    output[start_idx:end_idx] = segment
        
        return output
    
    def apply_reverb_simple(self, audio: np.ndarray, 
                            decay: float = 0.3, 
                            delay_ms: float = 30) -> np.ndarray:
        """
        Applique une réverb simple.
        
        Args:
            audio: Signal audio
            decay: Facteur de décroissance (0-1)
            delay_ms: Délai en millisecondes
            
        Returns:
            np.ndarray: Audio avec réverb
        """
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        
        output = audio.copy()
        
        # Ajouter plusieurs échos avec décroissance
        for i in range(1, 6):
            delay = delay_samples * i
            amplitude = decay ** i
            
            if delay < len(audio):
                output[delay:] += audio[:-delay] * amplitude
        
        # Normaliser pour éviter la saturation
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val * 0.95
        
        return output
    
    def apply_fade(self, audio: np.ndarray, 
                   fade_type: str = 'in',
                   duration_sec: float = 2.0,
                   curve: str = 'exponential') -> np.ndarray:
        """
        Applique un fade in ou out.
        
        Args:
            audio: Signal audio
            fade_type: 'in' ou 'out'
            duration_sec: Durée du fade en secondes
            curve: 'linear', 'exponential', 's_curve'
            
        Returns:
            np.ndarray: Audio avec fade
        """
        fade_samples = int(duration_sec * self.sample_rate)
        fade_samples = min(fade_samples, len(audio))
        
        # Créer la courbe de fade
        t = np.linspace(0, 1, fade_samples)
        
        if curve == 'linear':
            fade_curve = t
        elif curve == 'exponential':
            fade_curve = t ** 2
        elif curve == 's_curve':
            fade_curve = (1 - np.cos(t * np.pi)) / 2
        else:
            fade_curve = t
        
        if fade_type == 'out':
            fade_curve = 1 - fade_curve
        
        output = audio.copy()
        
        if fade_type == 'in':
            output[:fade_samples] *= fade_curve
        else:
            output[-fade_samples:] *= fade_curve
        
        return output
    
    def apply_echo(self, audio: np.ndarray, 
                   delay_ms: float = 250,
                   feedback: float = 0.4,
                   mix: float = 0.3) -> np.ndarray:
        """
        Applique un effet d'écho.
        
        Args:
            audio: Signal audio
            delay_ms: Délai en millisecondes
            feedback: Quantité de feedback (0-1)
            mix: Mélange dry/wet (0-1)
            
        Returns:
            np.ndarray: Audio avec écho
        """
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        
        output = np.zeros(len(audio) + delay_samples * 5)
        output[:len(audio)] = audio
        
        # Appliquer l'écho avec feedback
        for i in range(5):
            start = delay_samples * (i + 1)
            amplitude = feedback ** (i + 1)
            
            if start < len(output):
                end = min(start + len(audio), len(output))
                audio_len = end - start
                output[start:end] += audio[:audio_len] * amplitude
        
        # Tronquer et mixer
        output = output[:len(audio)]
        result = (1 - mix) * audio + mix * output
        
        # Normaliser
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val * 0.95
        
        return result