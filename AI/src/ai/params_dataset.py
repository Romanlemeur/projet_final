import numpy as np
import librosa
import pickle
import os
from pathlib import Path
import random

from src.analysis.feature_extractor import FeatureExtractor
from src.analysis.key_analyzer import KeyAnalyzer
from src.utils.config import SAMPLE_RATE


class TransitionParamsDataset:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.feature_extractor = FeatureExtractor(sample_rate)
        self.key_analyzer = KeyAnalyzer(sample_rate)
        self.data = []
        self.circle_of_fifths = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']

    def _extract_features(self, audio):
        features = self.feature_extractor.extract_all(audio)
        key_info = self.key_analyzer.detect_key(audio)
        
        try:
            tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
            beat_reg = 1.0 - np.std(np.diff(beat_times)) / (np.mean(np.diff(beat_times)) + 1e-8) if len(beat_times) > 2 else 0.5
            beat_reg = max(0, min(1, beat_reg))
        except:
            beat_reg = 0.5
        
        stft = np.abs(librosa.stft(audio))
        low_energy = np.mean(stft[:int(stft.shape[0] * 0.1), :])
        mid_energy = np.mean(stft[int(stft.shape[0] * 0.1):int(stft.shape[0] * 0.5), :])
        high_energy = np.mean(stft[int(stft.shape[0] * 0.5):, :])
        total = low_energy + mid_energy + high_energy + 1e-8
        
        rms = librosa.feature.rms(y=audio)[0]
        rms_var = np.std(rms) / (np.mean(rms) + 1e-8)
        
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        onset_density = np.sum(onset_env > np.mean(onset_env) * 1.5) / len(onset_env)
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
        
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        mfcc_mean = np.mean(mfcc[1:4])
        has_vocals = 1.0 if (mfcc_mean > -10 and spectral_centroid > 1500) else 0.0
        
        key = key_info['key']
        key_pos = self.circle_of_fifths.index(key) / 12.0 if key in self.circle_of_fifths else 0.5
        
        return {
            'features_array': np.array([
                features['bpm'] / 200.0,
                features['energy'],
                key_pos,
                1.0 if key_info['mode'] == 'major' else 0.0,
                key_info['confidence'],
                low_energy / total,
                mid_energy / total,
                high_energy / total,
                min(spectral_centroid / 5000.0, 1.0),
                min(spectral_rolloff / 10000.0, 1.0),
                min(spectral_flatness * 10, 1.0),
                beat_reg,
                min(rms_var, 1.0),
                min(onset_density * 5, 1.0)
            ], dtype=np.float32),
            'bpm': features['bpm'],
            'energy': features['energy'],
            'key': key,
            'mode': key_info['mode'],
            'has_vocals': has_vocals,
            'low_ratio': low_energy / total,
            'mid_ratio': mid_energy / total,
            'high_ratio': high_energy / total,
            'beat_regularity': beat_reg
        }

    def _key_compatibility(self, key1, mode1, key2, mode2):
        if key1 not in self.circle_of_fifths or key2 not in self.circle_of_fifths:
            return 0.5
        idx1 = self.circle_of_fifths.index(key1)
        idx2 = self.circle_of_fifths.index(key2)
        dist = min(abs(idx1 - idx2), 12 - abs(idx1 - idx2))
        if dist == 0:
            return 1.0
        elif dist == 1:
            return 0.9
        elif dist == 2:
            return 0.7
        elif mode1 != mode2 and dist <= 3:
            return 0.6
        else:
            return max(0.2, 1.0 - dist * 0.1)

    def _compute_ideal_params(self, f1, f2):
        bpm1, bpm2 = f1['bpm'], f2['bpm']
        energy1, energy2 = f1['energy'], f2['energy']
        vocals1, vocals2 = f1['has_vocals'], f2['has_vocals']
        low1, low2 = f1['low_ratio'], f2['low_ratio']
        mid1, mid2 = f1['mid_ratio'], f2['mid_ratio']
        high1, high2 = f1['high_ratio'], f2['high_ratio']
        beat_reg1, beat_reg2 = f1['beat_regularity'], f2['beat_regularity']
        
        key_compat = self._key_compatibility(f1['key'], f1['mode'], f2['key'], f2['mode'])
        bpm_diff = abs(bpm1 - bpm2) / 200.0
        energy_diff = abs(energy1 - energy2)
        similarity = (key_compat + (1 - bpm_diff) + (1 - energy_diff)) / 3
        
        if similarity > 0.7:
            mix_style = 0
            transition_beats = 0.5
        elif energy2 > energy1 + 0.2:
            mix_style = 1
            transition_beats = 0.7
        elif vocals1 > 0.5 or vocals2 > 0.5:
            mix_style = 2
            transition_beats = 0.3
        else:
            mix_style = 3
            transition_beats = 0.5
        
        low_eq_1_end = -0.8 if low2 > 0.3 else -0.3
        mid_eq_1_end = -0.5 if vocals2 > 0.5 else 0.0
        high_eq_1_end = -0.3
        
        low_eq_2_start = -1.0 if low1 > 0.3 else -0.5
        mid_eq_2_start = -0.3 if vocals1 > 0.5 else 0.0
        high_eq_2_start = -0.5
        
        crossfade_type = 0 if similarity > 0.8 else (1 if energy2 > energy1 else 2)
        crossfade_pos = 0.5 if similarity > 0.7 else (0.4 if energy2 > energy1 else 0.6)
        
        cue_out = 0.85 if beat_reg1 > 0.7 else 0.75
        cue_in = 0.05 if beat_reg2 > 0.7 else 0.1
        
        align_beat = beat_reg1 * 0.5 + beat_reg2 * 0.5
        align_bar = 1.0 if (beat_reg1 > 0.8 and beat_reg2 > 0.8) else 0.5
        
        eq_swap = 0.5 if similarity > 0.7 else 0.4
        bass_swap = 0.5 if low1 < low2 else 0.6
        
        filter_sweep = 0.7 if mix_style == 1 else (0.3 if similarity > 0.7 else 0.5)
        filter_res = 0.3 + (1 - similarity) * 0.3
        
        tension = 0 if similarity > 0.8 else (1 if mix_style == 1 else (2 if vocals1 > 0.5 else 3))
        
        duck_v1 = 0.7 if (vocals1 > 0.5 and vocals2 > 0.5) else 0.0
        duck_v2 = 0.3 if (vocals1 > 0.5 and vocals2 > 0.5) else 0.0
        
        energy_dir = 0.7 if energy2 > energy1 else (0.3 if energy1 > energy2 else 0.5)
        
        return np.array([
            (low_eq_1_end + 1.0) / 1.5,
            (mid_eq_1_end + 1.0) / 1.5,
            (high_eq_1_end + 1.0) / 1.5,
            (low_eq_2_start + 1.0) / 1.5,
            (mid_eq_2_start + 1.0) / 1.5,
            (high_eq_2_start + 1.0) / 1.5,
            0.7 if energy1 > energy2 else 0.5,
            0.7 if energy2 > energy1 else 0.5,
            crossfade_type / 3.0,
            (crossfade_pos - 0.3) / 0.4,
            (cue_out - 0.6) / 0.35,
            cue_in / 0.2,
            align_beat,
            align_bar,
            transition_beats,
            (eq_swap - 0.3) / 0.4,
            (bass_swap - 0.4) / 0.3,
            mix_style / 4.0,
            filter_sweep,
            (filter_res - 0.1) / 0.6,
            tension / 4.0,
            duck_v1,
            duck_v2,
            energy_dir
        ], dtype=np.float32)

    def build_from_folder(self, music_folder, max_songs=2000, max_pairs=10000):
        print(f"Scan: {music_folder}")
        audio_files = list(Path(music_folder).rglob("*.mp3"))
        print(f"Fichiers: {len(audio_files)}")
        
        random.shuffle(audio_files)
        audio_files = audio_files[:max_songs]
        audio_data = []
        
        print(f"Analyse: {len(audio_files)} morceaux")
        for i, f in enumerate(audio_files):
            try:
                audio, _ = librosa.load(str(f), sr=self.sample_rate, mono=True, duration=30)
                if len(audio) > self.sample_rate * 5:
                    audio_data.append({'file': f.name, 'features': self._extract_features(audio)})
            except:
                pass
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(audio_files)}")
        
        print(f"Analyses: {len(audio_data)}")
        
        n_pairs = min(max_pairs, len(audio_data) * (len(audio_data) - 1) // 2)
        print(f"Creation: {n_pairs} paires")
        
        for i in range(n_pairs):
            idx1, idx2 = random.sample(range(len(audio_data)), 2)
            f1, f2 = audio_data[idx1]['features'], audio_data[idx2]['features']
            self.data.append({
                'input': np.concatenate([f1['features_array'], f2['features_array']]),
                'target': self._compute_ideal_params(f1, f2)
            })
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{n_pairs}")
        
        print(f"Dataset: {len(self.data)} paires")

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.data = pickle.load(f)

    def get_batch(self, batch_size=32):
        indices = random.sample(range(len(self.data)), min(batch_size, len(self.data)))
        return (
            np.array([self.data[i]['input'] for i in indices]),
            np.array([self.data[i]['target'] for i in indices])
        )

    def __len__(self):
        return len(self.data)
