"""Configuration globale du projet."""

SAMPLE_RATE = 44100  
HOP_LENGTH = 512    
N_FFT = 2048        

SUPPORTED_FORMATS = ['.mp3', '.wav', '.flac', '.ogg']

DEFAULT_TRANSITION_DURATION = 8  # secondes
MIN_TRANSITION_DURATION = 4
MAX_TRANSITION_DURATION = 16