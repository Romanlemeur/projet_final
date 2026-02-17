"""Dataset avec analyse des points de coupe optimaux."""

import numpy as np
import librosa
import pickle
import os
from pathlib import Path
import random

from src.analysis.feature_extractor import FeatureExtractor
from src.analysis.key_analyzer import KeyAnalyzer
from src.analysis.beat_detector import BeatDetector
from src.utils.config import SAMPLE_RATE


class TransitionParamsDataset:
    """
    Dataset qui génère des paramètres de transition "idéaux"
    incluant les points de coupe optimaux.
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.feature_extractor = FeatureExtractor(sample_rate)
        self.key_analyzer = KeyAnalyzer(sample_rate)
        self.beat_detector = BeatDetector(sample_rate)
        self.data = []
        
        self.circle_of_fifths = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']
    
    def _extract_features(self, audio: np.ndarray) -> dict:
        """Extrait les features complètes incluant l'analyse des beats."""
        
        features = self.feature_extractor.extract_all(audio)
        key_info = self.key_analyzer.detect_key(audio)
        
        # Analyse des beats
        try:
            tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
            
            # Calculer la force des beats
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            beat_strengths = onset_env[beat_frames] if len(beat_frames) > 0 else np.array([0.5])
        except:
            tempo = features['bpm']
            beat_times = np.array([])
            beat_strengths = np.array([0.5])
        
        # Analyse de l'énergie par segment
        duration = len(audio) / self.sample_rate
        n_segments = 10
        segment_len = len(audio) // n_segments
        
        energy_profile = []
        for i in range(n_segments):
            start = i * segment_len
            end = (i + 1) * segment_len
            seg_energy = np.sqrt(np.mean(audio[start:end] ** 2))
            energy_profile.append(seg_energy)
        
        energy_profile = np.array(energy_profile)
        energy_profile = energy_profile / (np.max(energy_profile) + 1e-8)
        
        # Trouver les meilleurs points de coupe (où l'énergie est basse)
        best_outro_segment = np.argmin(energy_profile[5:]) + 5  # Chercher dans la 2ème moitié
        best_intro_segment = np.argmin(energy_profile[:5])  # Chercher dans la 1ère moitié
        
        # Features spectrales
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate))
        zero_crossing = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # Régularité des beats (pour savoir si on peut couper sur un beat)
        if len(beat_times) > 2:
            beat_intervals = np.diff(beat_times)
            beat_regularity = 1.0 - np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-8)
            beat_regularity = max(0, min(1, beat_regularity))
        else:
            beat_regularity = 0.5
        
        # Normaliser
        bpm_norm = features['bpm'] / 200.0
        energy_norm = features['energy']
        
        key = key_info['key']
        key_pos = self.circle_of_fifths.index(key) / 12.0 if key in self.circle_of_fifths else 0.5
        mode_binary = 1.0 if key_info['mode'] == 'major' else 0.0
        
        return {
            'features_array': np.array([
                bpm_norm,
                energy_norm,
                key_pos,
                mode_binary,
                key_info['confidence'],
                min(spectral_centroid / 5000.0, 1.0),
                min(spectral_rolloff / 10000.0, 1.0),
                min(zero_crossing * 10, 1.0),
                features['duration'] / 300.0,
                np.std(audio),
                beat_regularity,
                best_outro_segment / 10.0,
                best_intro_segment / 10.0,
                np.mean(beat_strengths) if len(beat_strengths) > 0 else 0.5
            ], dtype=np.float32),
            'bpm': features['bpm'],
            'energy': features['energy'],
            'beat_regularity': beat_regularity,
            'energy_profile': energy_profile,
            'best_outro': best_outro_segment / 10.0,
            'best_intro': best_intro_segment / 10.0,
            'spectral_centroid': spectral_centroid / 5000.0,
            'key_pos': key_pos,
            'mode': mode_binary
        }
    
    def _compute_ideal_params(self, feat1: dict, feat2: dict) -> np.ndarray:
        """
        Calcule les paramètres de transition "idéaux" incluant les points de coupe.
        """
        
        f1 = feat1['features_array']
        f2 = feat2['features_array']
        
        bpm1, energy1, key1, mode1 = f1[0], f1[1], f1[2], f1[3]
        bpm2, energy2, key2, mode2 = f2[0], f2[1], f2[2], f2[3]
        
        centroid1, centroid2 = f1[5], f2[5]
        beat_reg1, beat_reg2 = f1[10], f2[10]
        best_outro1, best_intro2 = f1[11], f2[12]
        
        # === CALCULS DE BASE ===
        bpm_diff = abs(bpm1 - bpm2)
        energy_diff = abs(energy1 - energy2)
        key_diff = abs(key1 - key2)
        similarity = 1 - (bpm_diff + energy_diff + key_diff) / 3
        
        # === POINTS DE COUPE (NOUVEAU) ===
        
        # Track 1 - Point de coupe
        # Si l'énergie baisse vers la fin, couper plus tard
        # Si les beats sont réguliers, préférer couper sur un beat
        track1_cut_position = 0.7 + best_outro1 * 0.2  # Utiliser l'analyse d'énergie
        track1_cut_on_beat = beat_reg1  # Plus régulier = plus on coupe sur un beat
        track1_cut_on_bar = beat_reg1 * 0.8  # Préférence mesure légèrement plus faible
        track1_fade_before = 0.2 + (1 - similarity) * 0.3  # Plus long si morceaux différents
        
        # Track 2 - Point de départ
        # Si l'énergie est basse au début, commencer tôt
        # Sinon, chercher un meilleur point d'entrée
        track2_start_position = best_intro2 * 0.3  # Utiliser l'analyse d'énergie
        track2_start_on_beat = beat_reg2
        track2_start_on_bar = beat_reg2 * 0.8
        track2_fade_after = 0.2 + (1 - similarity) * 0.3
        
        # === STRUCTURE ===
        if similarity > 0.7:
            phase1_dur, phase2_dur, phase3_dur = 0.30, 0.30, 0.40
        elif similarity > 0.4:
            phase1_dur, phase2_dur, phase3_dur = 0.35, 0.35, 0.30
        else:
            phase1_dur, phase2_dur, phase3_dur = 0.30, 0.45, 0.25
        
        total_dur_factor = 0.8 + (1 - similarity) * 0.4
        overlap = 0.15 + similarity * 0.1
        
        # === FILTRES P1 ===
        p1_filter_start = 0.7 + centroid1 * 0.3
        p1_filter_end = 0.1 + similarity * 0.2
        p1_resonance = 0.3 + (1 - similarity) * 0.3
        p1_curve = 0.5 + (1 - similarity) * 0.3
        
        # === FILTRES P3 ===
        p3_filter_start = 0.15 + (1 - similarity) * 0.1
        p3_filter_end = 0.6 + centroid2 * 0.4
        p3_resonance = 0.2 + centroid2 * 0.2
        p3_curve = 0.4 + similarity * 0.2
        
        # === EFFETS ===
        reverb_amount = 0.2 + (1 - similarity) * 0.4
        reverb_decay = 0.3 + (1 - similarity) * 0.3
        echo_amount = 0.1 + (1 - similarity) * 0.3
        echo_delay = 0.3 + bpm1 * 0.2
        echo_feedback = 0.2 + (1 - similarity) * 0.2
        use_riser = 1.0 if (1 - similarity) > 0.4 else 0.0
        
        # === MIX ===
        drums_vol, harmonic_vol, bass_vol = 0.35, 0.45, 0.20
        bass_swap = 0.5 + energy_diff * 0.2
        crossfade_curve = similarity * 0.5 + 0.25
        
        if energy2 > energy1:
            energy_curve = 0.6 + (energy2 - energy1) * 0.4
        else:
            energy_curve = 0.4 - (energy1 - energy2) * 0.2
        
        # === STYLE ===
        if similarity > 0.7 and energy1 > 0.5:
            style_smooth, style_dramatic, style_ambient = 0.7, 0.2, 0.1
        elif (1 - similarity) > 0.5:
            style_smooth, style_dramatic, style_ambient = 0.2, 0.6, 0.2
        else:
            style_smooth, style_dramatic, style_ambient = 0.4, 0.3, 0.3
        
        brightness_target = (centroid1 + centroid2) / 2
        tension = (1 - similarity) * 0.7 + bpm_diff * 0.3
        
        return np.array([
            # Points de coupe (8)
            track1_cut_position, track1_cut_on_beat, track1_cut_on_bar, track1_fade_before,
            track2_start_position, track2_start_on_beat, track2_start_on_bar, track2_fade_after,
            # Structure (5)
            phase1_dur, phase2_dur, phase3_dur, total_dur_factor, overlap,
            # Filtres P1 (4)
            p1_filter_start, p1_filter_end, p1_resonance, p1_curve,
            # Filtres P3 (4)
            p3_filter_start, p3_filter_end, p3_resonance, p3_curve,
            # Effets (6)
            reverb_amount, reverb_decay, echo_amount, echo_delay, echo_feedback, use_riser,
            # Mix (6)
            drums_vol, harmonic_vol, bass_vol, bass_swap, crossfade_curve, energy_curve,
            # Style (5)
            style_smooth, style_dramatic, style_ambient, brightness_target, tension
        ], dtype=np.float32)
    
    def build_from_folder(self, music_folder: str, max_samples: int = 800):
        print(f"📂 Scan: {music_folder}")
        
        audio_files = []
        for ext in ['*.mp3', '*.wav', '*.flac']:
            audio_files.extend(Path(music_folder).glob(ext))
        
        audio_files = list(audio_files)
        print(f"  Fichiers: {len(audio_files)}")
        
        if len(audio_files) < 2:
            print("  ⚠️ Il faut au moins 2 fichiers !")
            return
        
        print("🎵 Analyse approfondie des morceaux...")
        audio_data = []
        
        for f in audio_files[:50]:  # Analyser jusqu'à 50 fichiers
            try:
                audio, _ = librosa.load(str(f), sr=self.sample_rate, mono=True, duration=90)
                if len(audio) > self.sample_rate * 15:
                    features = self._extract_features(audio)
                    audio_data.append({'file': f.name, 'features': features})
                    print(f"  ✓ {f.name} (BPM: {features['bpm']:.0f}, Energy: {features['energy']:.2f})")
            except Exception as e:
                print(f"  ✗ {f.name}: {e}")
        
        print(f"\n📊 Analysés: {len(audio_data)}")
        print(f"🔨 Création des échantillons...")
        
        for i in range(min(max_samples, len(audio_data) * (len(audio_data) - 1))):
            idx1, idx2 = random.sample(range(len(audio_data)), 2)
            
            feat1 = audio_data[idx1]['features']
            feat2 = audio_data[idx2]['features']
            
            # Input: features des 2 morceaux (14 + 14 = 28)
            input_features = np.concatenate([
                feat1['features_array'],
                feat2['features_array']
            ])
            
            # Target: paramètres idéaux (38)
            target_params = self._compute_ideal_params(feat1, feat2)
            
            self.data.append({'input': input_features, 'target': target_params})
        
        print(f"✅ Dataset: {len(self.data)} échantillons")
    
    def augment(self):
        """Augmentation intensive."""
        print("🔄 Augmentation...")
        original = len(self.data)
        augmented = []
        
        for sample in self.data:
            # 1. Inverser
            input_inv = np.concatenate([sample['input'][14:], sample['input'][:14]])
            target_inv = sample['target'].copy()
            # Inverser les points de coupe
            target_inv[0:4], target_inv[4:8] = target_inv[4:8].copy(), target_inv[0:4].copy()
            augmented.append({'input': input_inv, 'target': target_inv})
            
            # 2-4. Bruit
            for noise_level in [0.02, 0.04, 0.06]:
                noise = np.random.randn(*sample['input'].shape) * noise_level
                augmented.append({
                    'input': np.clip(sample['input'] + noise, 0, 1).astype(np.float32),
                    'target': sample['target']
                })
            
            # 5. Variation targets
            target_var = sample['target'] + np.random.randn(*sample['target'].shape) * 0.03
            augmented.append({
                'input': sample['input'],
                'target': np.clip(target_var, 0, 1).astype(np.float32)
            })
        
        self.data.extend(augmented)
        print(f"  {original} → {len(self.data)}")
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"💾 Sauvé: {path}")
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"📂 Chargé: {len(self.data)}")
    
    def get_batch(self, batch_size: int = 32):
        indices = random.sample(range(len(self.data)), min(batch_size, len(self.data)))
        inputs = np.array([self.data[i]['input'] for i in indices])
        targets = np.array([self.data[i]['target'] for i in indices])
        return inputs, targets
    
    def __len__(self):
        return len(self.data)