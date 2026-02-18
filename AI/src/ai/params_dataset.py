import numpy as np
import librosa
import pickle
import os
from pathlib import Path
import random
from scipy.signal import medfilt
from scipy.ndimage import uniform_filter1d

from src.analysis.feature_extractor import FeatureExtractor
from src.analysis.key_analyzer import KeyAnalyzer
from src.utils.config import SAMPLE_RATE


CAMELOT_WHEEL = {
    'C': '8B', 'Am': '8A', 'G': '9B', 'Em': '9A', 'D': '10B', 'Bm': '10A',
    'A': '11B', 'F#m': '11A', 'E': '12B', 'C#m': '12A', 'B': '1B', 'G#m': '1A',
    'F#': '2B', 'D#m': '2A', 'Db': '3B', 'Bbm': '3A', 'Ab': '4B', 'Fm': '4A',
    'Eb': '5B', 'Cm': '5A', 'Bb': '6B', 'Gm': '6A', 'F': '7B', 'Dm': '7A'
}


class TransitionParamsDataset:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.feature_extractor = FeatureExtractor(sample_rate)
        self.key_analyzer = KeyAnalyzer(sample_rate)
        self.data = []
        self.circle_of_fifths = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']
        self.hop_length = 512

    def _detect_vocals_score(self, audio):
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13, hop_length=self.hop_length)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
        flatness = librosa.feature.spectral_flatness(y=audio, hop_length=self.hop_length)[0]
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate, hop_length=self.hop_length)
        
        centroid_norm = centroid / (self.sample_rate / 2)
        mfcc_var = np.var(mfcc[1:6], axis=0)
        mfcc_energy = np.mean(np.abs(mfcc[1:5]), axis=0)
        contrast_mid = np.mean(contrast[2:5], axis=0)
        
        score = np.zeros(len(centroid))
        score += (centroid_norm > 0.03) * (centroid_norm < 0.28) * 0.24
        score += (flatness < 0.18) * 0.18
        score += (mfcc_var > np.percentile(mfcc_var, 20)) * 0.20
        score += (mfcc_energy > np.percentile(mfcc_energy, 22)) * 0.16
        score += (contrast_mid > np.percentile(contrast_mid, 28)) * 0.22
        
        score = medfilt(score, kernel_size=17)
        return np.mean(score)

    def _get_breakdown_score(self, audio):
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        rms_smooth = uniform_filter1d(rms, size=25)
        
        try:
            _, percussive = librosa.effects.hpss(audio)
            perc_rms = librosa.feature.rms(y=percussive, hop_length=self.hop_length)[0]
            perc_ratio = np.mean(perc_rms) / (np.mean(rms_smooth) + 1e-8)
        except:
            perc_ratio = 0.5
        
        energy_stability = 1.0 - min(np.std(rms_smooth) / (np.mean(rms_smooth) + 1e-8), 1.0)
        
        return (1 - perc_ratio) * 0.5 + energy_stability * 0.5

    def _get_camelot(self, key, mode):
        key_str = key + 'm' if mode == 'minor' else key
        for k, cam in CAMELOT_WHEEL.items():
            if k.lower() == key_str.lower() or k.lower() == key.lower():
                return cam
        return '8A' if mode == 'minor' else '8B'

    def _camelot_distance(self, cam1, cam2):
        if cam1 == cam2:
            return 0
        num1, num2 = int(cam1[:-1]), int(cam2[:-1])
        letter1, letter2 = cam1[-1], cam2[-1]
        
        if num1 == num2 and letter1 != letter2:
            return 0.5
        
        dist = min(abs(num1 - num2), 12 - abs(num1 - num2))
        return dist

    def _extract_features(self, audio):
        features = self.feature_extractor.extract_all(audio)
        key_info = self.key_analyzer.detect_key(audio)
        
        try:
            tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
            beat_reg = 1.0 - np.std(np.diff(beat_times)) / (np.mean(np.diff(beat_times)) + 1e-8) if len(beat_times) > 2 else 0.5
            beat_reg = max(0, min(1, beat_reg))
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0])
        except:
            beat_reg = 0.5
            tempo = features['bpm']
        
        stft = np.abs(librosa.stft(audio))
        low_e = np.mean(stft[:int(stft.shape[0] * 0.1), :])
        mid_e = np.mean(stft[int(stft.shape[0] * 0.1):int(stft.shape[0] * 0.5), :])
        high_e = np.mean(stft[int(stft.shape[0] * 0.5):, :])
        total = low_e + mid_e + high_e + 1e-8
        
        rms = librosa.feature.rms(y=audio)[0]
        rms_var = np.std(rms) / (np.mean(rms) + 1e-8)
        
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        onset_density = np.sum(onset_env > np.mean(onset_env) * 1.5) / len(onset_env)
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
        
        vocal_score = self._detect_vocals_score(audio)
        breakdown_score = self._get_breakdown_score(audio)
        
        camelot = self._get_camelot(key_info['key'], key_info['mode'])
        
        key = key_info['key']
        key_pos = self.circle_of_fifths.index(key) / 12.0 if key in self.circle_of_fifths else 0.5
        
        return {
            'features_array': np.array([
                tempo / 200.0,
                features['energy'],
                key_pos,
                1.0 if key_info['mode'] == 'major' else 0.0,
                key_info['confidence'],
                low_e / total,
                mid_e / total,
                high_e / total,
                min(spectral_centroid / 5000.0, 1.0),
                min(spectral_rolloff / 10000.0, 1.0),
                min(spectral_flatness * 10, 1.0),
                beat_reg,
                min(rms_var, 1.0),
                min(onset_density * 5, 1.0)
            ], dtype=np.float32),
            'bpm': tempo,
            'energy': features['energy'],
            'key': key_info['key'],
            'mode': key_info['mode'],
            'camelot': camelot,
            'key_confidence': key_info['confidence'],
            'low_ratio': low_e / total,
            'mid_ratio': mid_e / total,
            'high_ratio': high_e / total,
            'beat_regularity': beat_reg,
            'vocal_presence': vocal_score,
            'breakdown_score': breakdown_score
        }

    def _compute_ideal_params(self, f1, f2):
        bpm1, bpm2 = f1['bpm'], f2['bpm']
        energy1, energy2 = f1['energy'], f2['energy']
        vocals1, vocals2 = f1['vocal_presence'], f2['vocal_presence']
        breakdown1, breakdown2 = f1['breakdown_score'], f2['breakdown_score']
        low1, low2 = f1['low_ratio'], f2['low_ratio']
        beat_reg1, beat_reg2 = f1['beat_regularity'], f2['beat_regularity']
        
        cam_dist = self._camelot_distance(f1['camelot'], f2['camelot'])
        key_compat = max(0, 1.0 - cam_dist * 0.15)
        
        bpm_diff = abs(bpm1 - bpm2) / max(bpm1, bpm2)
        energy_diff = abs(energy1 - energy2)
        similarity = (key_compat * 0.4 + (1 - bpm_diff) * 0.35 + (1 - energy_diff) * 0.25)
        
        both_vocals = vocals1 > 0.38 and vocals2 > 0.38
        
        if both_vocals:
            duck_v1 = 0.72
            duck_v2 = 0.55
            mid_eq_1 = -0.75
            mid_eq_2 = -0.55
        else:
            duck_v1 = vocals1 * 0.32
            duck_v2 = vocals2 * 0.32
            mid_eq_1 = -0.22 if vocals1 > 0.38 else 0.0
            mid_eq_2 = -0.22 if vocals2 > 0.38 else 0.0
        
        good_bd1 = breakdown1 > 0.58 and vocals1 < 0.32
        good_bd2 = breakdown2 > 0.58 and vocals2 < 0.32
        
        if good_bd1 and good_bd2:
            cue_out = 0.62
            cue_in = 0.12
        elif good_bd1:
            cue_out = 0.58
            cue_in = 0.06
        elif good_bd2:
            cue_out = 0.78
            cue_in = 0.14
        else:
            cue_out = 0.76
            cue_in = 0.06
        
        if similarity > 0.72:
            mix_style = 0
            transition_beats = 32
            crossfade_type = 1
        elif energy2 > energy1 + 0.12:
            mix_style = 3
            transition_beats = 48
            crossfade_type = 2
        elif bpm_diff > 0.08:
            mix_style = 2
            transition_beats = 24
            crossfade_type = 0
        else:
            mix_style = 1
            transition_beats = 32
            crossfade_type = 1
        
        low_eq_1 = -0.92 if low2 > 0.32 else -0.52
        low_eq_2 = -1.05 if low1 > 0.32 else -0.62
        high_eq_1 = -0.32
        high_eq_2 = -0.42
        
        bass_swap = 0.42 if similarity > 0.7 else (0.32 if energy2 > energy1 else 0.52)
        eq_swap = 0.38 if similarity > 0.7 else 0.28
        
        filter_sweep = 0.82 if mix_style == 3 else (0.18 if similarity > 0.82 else 0.48)
        filter_res = 0.14 + (1 - similarity) * 0.38
        
        if similarity > 0.85:
            tension = 0
        elif mix_style == 3:
            tension = 3
        elif vocals1 > 0.48 or vocals2 > 0.48:
            tension = 2
        else:
            tension = 1
        
        energy_dir = 0.78 if energy2 > energy1 else (0.22 if energy1 > energy2 else 0.5)
        
        return np.array([
            (low_eq_1 + 1.0) / 1.5,
            (mid_eq_1 + 1.0) / 1.5,
            (high_eq_1 + 1.0) / 1.5,
            (low_eq_2 + 1.0) / 1.5,
            (mid_eq_2 + 1.0) / 1.5,
            (high_eq_2 + 1.0) / 1.5,
            0.65 if energy1 > energy2 else 0.55,
            0.65 if energy2 > energy1 else 0.55,
            crossfade_type / 3.0,
            0.5,
            (cue_out - 0.6) / 0.35,
            cue_in / 0.2,
            (beat_reg1 + beat_reg2) / 2,
            1.0 if (beat_reg1 > 0.7 and beat_reg2 > 0.7) else 0.6,
            (transition_beats - 16) / 48,
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

    def build_from_folder(self, music_folder, max_songs=200, max_pairs=2000):
        print(f"Scanning: {music_folder}")
        
        audio_files = []
        for ext in ['*.mp3', '*.wav', '*.flac', '*.ogg', '*.m4a']:
            audio_files.extend(list(Path(music_folder).rglob(ext)))
        
        print(f"Found: {len(audio_files)} files")
        
        random.shuffle(audio_files)
        audio_files = audio_files[:max_songs]
        audio_data = []
        
        print(f"Analyzing: {len(audio_files)} tracks")
        
        for i, f in enumerate(audio_files):
            try:
                audio, _ = librosa.load(str(f), sr=self.sample_rate, mono=True, duration=30)
                if len(audio) > self.sample_rate * 5:
                    features = self._extract_features(audio)
                    audio_data.append({
                        'file': f.name,
                        'features': features
                    })
            except Exception as e:
                pass
            
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(audio_files)} ({len(audio_data)} valid)")
        
        print(f"Valid tracks: {len(audio_data)}")
        
        if len(audio_data) < 2:
            print("Error: Not enough valid tracks")
            return
        
        n_pairs = min(max_pairs, len(audio_data) * (len(audio_data) - 1) // 2)
        print(f"Generating: {n_pairs} pairs")
        
        for i in range(n_pairs):
            idx1, idx2 = random.sample(range(len(audio_data)), 2)
            f1, f2 = audio_data[idx1]['features'], audio_data[idx2]['features']
            
            self.data.append({
                'input': np.concatenate([f1['features_array'], f2['features_array']]),
                'target': self._compute_ideal_params(f1, f2)
            })
            
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{n_pairs}")
        
        print(f"Dataset size: {len(self.data)} samples")

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"Saved: {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Loaded: {len(self.data)} samples")

    def get_batch(self, batch_size=32):
        indices = random.sample(range(len(self.data)), min(batch_size, len(self.data)))
        return (
            np.array([self.data[i]['input'] for i in indices]),
            np.array([self.data[i]['target'] for i in indices])
        )

    def __len__(self):
        return len(self.data)
