"""Module de séparation des éléments audio."""

import numpy as np
import librosa
from scipy.ndimage import median_filter
from src.utils.config import SAMPLE_RATE, HOP_LENGTH, N_FFT


class AudioSeparator:
    """Sépare les composantes d'un signal audio (percussions, harmoniques)."""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.hop_length = HOP_LENGTH
        self.n_fft = N_FFT
    
    def separate_harmonic_percussive(self, audio: np.ndarray) -> tuple:
        harmonic, percussive = librosa.effects.hpss(audio)
        
        return harmonic, percussive
    
    def extract_bass(self, audio: np.ndarray, cutoff: int = 250) -> np.ndarray:
        
        from scipy.signal import butter, filtfilt
        
        # Filtre passe-bas
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        b, a = butter(4, normalized_cutoff, btype='low')
        
        bass = filtfilt(b, a, audio)
        
        return bass
    
    def extract_highs(self, audio: np.ndarray, cutoff: int = 2000) -> np.ndarray:
        
        from scipy.signal import butter, filtfilt
        
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        b, a = butter(4, normalized_cutoff, btype='high')
        
        highs = filtfilt(b, a, audio)
        
        return highs
    
    def extract_mids(self, audio: np.ndarray, low: int = 250, high: int = 2000) -> np.ndarray:
        
        from scipy.signal import butter, filtfilt
        
        nyquist = self.sample_rate / 2
        low_norm = low / nyquist
        high_norm = high / nyquist
        b, a = butter(4, [low_norm, high_norm], btype='band')
        
        mids = filtfilt(b, a, audio)
        
        return mids
    
    def full_separation(self, audio: np.ndarray) -> dict:
        
        harmonic, percussive = self.separate_harmonic_percussive(audio)
        
        return {
            'full': audio,
            'harmonic': harmonic,
            'percussive': percussive,
            'bass': self.extract_bass(audio),
            'mids': self.extract_mids(audio),
            'highs': self.extract_highs(audio)
        }