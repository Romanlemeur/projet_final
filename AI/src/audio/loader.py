import os
import librosa
import numpy as np
from src.utils.config import SAMPLE_RATE, SUPPORTED_FORMATS


class AudioLoader:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate

    def load(self, file_path: str) -> tuple[np.ndarray, int]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier non trouvé: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(f"Format non supporté: {ext}. Formats acceptés: {SUPPORTED_FORMATS}")

        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)

        print(f"[OK] Audio charge: {os.path.basename(file_path)}")
        print(f"  - Durée: {len(audio) / sr:.2f} secondes")
        print(f"  - Sample rate: {sr} Hz")

        return audio, sr

    def load_segment(self, file_path: str, start: float, duration: float) -> tuple[np.ndarray, int]:
        audio, sr = librosa.load(
            file_path,
            sr=self.sample_rate,
            mono=True,
            offset=start,
            duration=duration
        )
        return audio, sr

    def get_duration(self, file_path: str) -> float:
        return librosa.get_duration(path=file_path)


def load_audio(file_path: str, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    loader = AudioLoader(sample_rate)
    return loader.load(file_path)
