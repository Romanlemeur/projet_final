import numpy as np
import librosa


class AudioCompatibilityScorer:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.hop_length = 512

    def _cosine_sim(self, a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.5
        return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))

    def _mel_fingerprint(self, audio, n_frames=30):
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=64, hop_length=self.hop_length
        )
        mel_db = librosa.power_to_db(mel + 1e-10)
        n = min(n_frames, mel_db.shape[1])
        return np.mean(mel_db[:, :n], axis=1)

    def score(self, audio1_end, audio2_start):
        min_len = max(int(0.3 * self.sample_rate), 256)
        if len(audio1_end) < min_len or len(audio2_start) < min_len:
            return {
                'total_score': 0.5, 'energy_continuity': 0.5,
                'spectral_match': 0.5, 'harmonic_coherence': 0.5,
                'brightness_match': 0.5,
            }

        # --- Energy continuity ---
        rms1 = librosa.feature.rms(y=audio1_end, hop_length=self.hop_length)[0]
        rms2 = librosa.feature.rms(y=audio2_start, hop_length=self.hop_length)[0]
        tail = max(5, len(rms1) // 6)
        head = max(5, len(rms2) // 6)
        end_level = float(np.mean(rms1[-tail:]))
        start_level = float(np.mean(rms2[:head]))
        denom = max(end_level, start_level, 1e-8)
        energy_diff = abs(end_level - start_level) / denom
        energy_continuity = max(0.0, 1.0 - energy_diff * 2.5)

        # --- Spectral match (mel fingerprint) ---
        tail_audio = audio1_end[-min(len(audio1_end), int(2 * self.sample_rate)):]
        head_audio = audio2_start[:min(len(audio2_start), int(2 * self.sample_rate))]
        mel_end = self._mel_fingerprint(tail_audio)
        mel_start = self._mel_fingerprint(head_audio)
        spectral_match = (self._cosine_sim(mel_end, mel_start) + 1.0) / 2.0

        # --- Harmonic coherence (chroma) ---
        chroma1 = np.mean(
            librosa.feature.chroma_stft(y=tail_audio, sr=self.sample_rate, hop_length=self.hop_length),
            axis=1
        )
        chroma2 = np.mean(
            librosa.feature.chroma_stft(y=head_audio, sr=self.sample_rate, hop_length=self.hop_length),
            axis=1
        )
        harmonic_coherence = (self._cosine_sim(chroma1, chroma2) + 1.0) / 2.0

        # --- Brightness match (spectral centroid) ---
        sc1 = float(np.mean(librosa.feature.spectral_centroid(y=tail_audio, sr=self.sample_rate)))
        sc2 = float(np.mean(librosa.feature.spectral_centroid(y=head_audio, sr=self.sample_rate)))
        sc_max = max(sc1, sc2, 1.0)
        brightness_match = max(0.0, 1.0 - abs(sc1 - sc2) / sc_max * 3.0)

        total = (
            energy_continuity  * 0.35 +
            spectral_match     * 0.30 +
            harmonic_coherence * 0.25 +
            brightness_match   * 0.10
        )

        return {
            'total_score':        round(float(total), 4),
            'energy_continuity':  round(float(energy_continuity), 4),
            'spectral_match':     round(float(spectral_match), 4),
            'harmonic_coherence': round(float(harmonic_coherence), 4),
            'brightness_match':   round(float(brightness_match), 4),
        }
