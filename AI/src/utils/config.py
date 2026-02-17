"""Configuration globale du projet."""

# Paramètres audio
SAMPLE_RATE = 44100  # Hz
HOP_LENGTH = 512     # Pour l'analyse
N_FFT = 2048         # Taille FFT

# Formats supportés
SUPPORTED_FORMATS = ['.mp3', '.wav', '.flac', '.ogg']

# Paramètres de transition
DEFAULT_TRANSITION_DURATION = 8  # secondes
MIN_TRANSITION_DURATION = 4
MAX_TRANSITION_DURATION = 16