"""Générateur de transition 100% piloté par l'IA - avec points de coupe intelligents."""

import numpy as np
import torch
import librosa
import os
from scipy.signal import butter, filtfilt

from src.ai.audio_separator import AudioSeparator
from src.analysis.feature_extractor import FeatureExtractor
from src.analysis.key_analyzer import KeyAnalyzer
from src.utils.config import SAMPLE_RATE

try:
    from src.ai.transition_params_model import TransitionParamsVAE
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False


class SmartTransitionGenerator:
    """
    Générateur de transition 100% piloté par l'IA.
    
    L'IA décide de TOUT:
    - OÙ couper le morceau 1
    - OÙ reprendre le morceau 2
    - La durée et structure
    - Les filtres et effets
    - Le style
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE,
                 model_path: str = "models/params_vae.pth"):
        self.sample_rate = sample_rate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.separator = AudioSeparator(sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate)
        self.key_analyzer = KeyAnalyzer(sample_rate)
        
        self.circle_of_fifths = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']
        
        self.model = None
        if MODEL_AVAILABLE and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_model(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model = TransitionParamsVAE(
                input_dim=checkpoint.get('input_dim', 28),
                latent_dim=checkpoint.get('latent_dim', 128),
                output_dim=checkpoint.get('output_dim', 38)
            ).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"  ✓ Modèle IA chargé (38 paramètres)")
        except Exception as e:
            print(f"  ✗ Erreur: {e}")
    
    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extrait les features complètes d'un morceau."""
        features = self.feature_extractor.extract_all(audio)
        key_info = self.key_analyzer.detect_key(audio)
        
        # Analyse beats
        try:
            tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            beat_strengths = onset_env[beat_frames] if len(beat_frames) > 0 else np.array([0.5])
            
            if len(beat_times) > 2:
                beat_intervals = np.diff(beat_times)
                beat_regularity = 1.0 - np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-8)
                beat_regularity = max(0, min(1, beat_regularity))
            else:
                beat_regularity = 0.5
        except:
            beat_regularity = 0.5
            beat_strengths = np.array([0.5])
        
        # Analyse énergie par segment
        n_segments = 10
        segment_len = len(audio) // n_segments
        energy_profile = []
        for i in range(n_segments):
            start, end = i * segment_len, (i + 1) * segment_len
            energy_profile.append(np.sqrt(np.mean(audio[start:end] ** 2)))
        energy_profile = np.array(energy_profile)
        energy_profile = energy_profile / (np.max(energy_profile) + 1e-8)
        
        best_outro = np.argmin(energy_profile[5:]) + 5
        best_intro = np.argmin(energy_profile[:5])
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate))
        zero_crossing = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        key = key_info['key']
        key_pos = self.circle_of_fifths.index(key) / 12.0 if key in self.circle_of_fifths else 0.5
        
        return np.array([
            features['bpm'] / 200.0,
            features['energy'],
            key_pos,
            1.0 if key_info['mode'] == 'major' else 0.0,
            key_info['confidence'],
            min(spectral_centroid / 5000.0, 1.0),
            min(spectral_rolloff / 10000.0, 1.0),
            min(zero_crossing * 10, 1.0),
            features['duration'] / 300.0,
            np.std(audio),
            beat_regularity,
            best_outro / 10.0,
            best_intro / 10.0,
            np.mean(beat_strengths) if len(beat_strengths) > 0 else 0.5
        ], dtype=np.float32)
    
    def _find_nearest_beat(self, audio: np.ndarray, target_time: float) -> float:
        """Trouve le beat le plus proche d'un temps donné."""
        try:
            _, beat_frames = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
            
            if len(beat_times) == 0:
                return target_time
            
            idx = np.argmin(np.abs(beat_times - target_time))
            return beat_times[idx]
        except:
            return target_time
    
    def _find_nearest_bar(self, audio: np.ndarray, target_time: float, beats_per_bar: int = 4) -> float:
        """Trouve le début de mesure le plus proche."""
        try:
            _, beat_frames = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
            
            if len(beat_times) < beats_per_bar:
                return target_time
            
            # Les mesures commencent tous les 4 beats
            bar_times = beat_times[::beats_per_bar]
            
            idx = np.argmin(np.abs(bar_times - target_time))
            return bar_times[idx]
        except:
            return target_time
    
    def _get_ai_params(self, audio1: np.ndarray, audio2: np.ndarray) -> dict:
        """L'IA prédit TOUS les paramètres."""
        features1 = self._extract_features(audio1)
        features2 = self._extract_features(audio2)
        
        if self.model is not None:
            input_features = np.concatenate([features1, features2])
            input_tensor = torch.FloatTensor(input_features).unsqueeze(0).to(self.device)
            params_tensor = self.model.predict(input_tensor)
            return self.model.get_params_dict(params_tensor)
        
        return self._default_params()
    
    def _default_params(self):
        return {
            'track1_cut_position': 0.8, 'track1_cut_on_beat': 0.7, 
            'track1_cut_on_bar': 0.5, 'track1_fade_before_cut': 0.3,
            'track2_start_position': 0.1, 'track2_start_on_beat': 0.7,
            'track2_start_on_bar': 0.5, 'track2_fade_after_start': 0.3,
            'phase1_duration': 0.33, 'phase2_duration': 0.34, 'phase3_duration': 0.33,
            'total_duration_factor': 1.0, 'overlap_duration': 0.2,
            'p1_filter_start': 0.8, 'p1_filter_end': 0.2, 
            'p1_filter_resonance': 0.3, 'p1_filter_curve': 0.5,
            'p3_filter_start': 0.2, 'p3_filter_end': 0.9,
            'p3_filter_resonance': 0.3, 'p3_filter_curve': 0.5,
            'reverb_amount': 0.3, 'reverb_decay': 0.4, 'echo_amount': 0.2,
            'echo_delay': 0.3, 'echo_feedback': 0.3, 'use_riser': 0.5,
            'drums_volume': 0.35, 'harmonic_volume': 0.45, 'bass_volume': 0.20,
            'bass_swap_point': 0.5, 'crossfade_curve': 0.5, 'energy_curve': 0.5,
            'style_smooth': 0.5, 'style_dramatic': 0.3, 'style_ambient': 0.2,
            'brightness_target': 0.5, 'tension_level': 0.5
        }
    
    def _apply_filter(self, audio, cutoff, filter_type='low'):
        nyquist = self.sample_rate / 2
        cutoff = max(80, min(cutoff, nyquist - 100))
        try:
            b, a = butter(4, cutoff / nyquist, btype=filter_type)
            return filtfilt(b, a, audio)
        except:
            return audio
    
    def _apply_reverb(self, audio, decay, delay_ms=40):
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        output = audio.copy()
        for i in range(1, 6):
            d = delay_samples * i
            if d < len(audio):
                output[d:] += audio[:-d] * (decay ** i)
        peak = np.max(np.abs(output))
        return output / peak * 0.95 if peak > 1 else output
    
    def _apply_echo(self, audio, delay_ms, feedback, mix):
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        output = audio.copy()
        for i in range(4):
            d = delay_samples * (i + 1)
            if d < len(audio):
                output[d:] += audio[:-d] * (feedback ** (i + 1))
        result = (1 - mix) * audio + mix * output
        peak = np.max(np.abs(result))
        return result / peak * 0.95 if peak > 1 else result
    
    def _filter_sweep(self, audio, start_freq, end_freq, curve_exp=1.0):
        n = len(audio)
        output = np.zeros(n)
        n_seg = 40
        seg_len = n // n_seg
        
        for i in range(n_seg):
            s, e = i * seg_len, min((i + 1) * seg_len, n)
            progress = (i / (n_seg - 1)) ** curve_exp
            freq = start_freq + (end_freq - start_freq) * progress
            if e > s:
                output[s:e] = self._apply_filter(audio[s:e], freq)
        
        return output
    
    def _create_riser(self, duration, intensity):
        n = int(duration * self.sample_rate)
        noise = np.random.randn(n)
        output = self._filter_sweep(noise, 200, 5000, 1.5)
        t = np.linspace(0, 1, n)
        return output * (t ** 2) * intensity
    
    def _resize(self, arr, length):
        if len(arr) >= length:
            return arr[:length]
        return np.pad(arr, (0, length - len(arr)))
    
    def generate_transition(self, audio1: np.ndarray, audio2: np.ndarray,
                            duration: float = 20.0,
                            style: str = 'auto') -> tuple:
        """
        Génère une transition où l'IA décide de TOUT.
        
        Returns:
            tuple: (transition_audio, cut_info)
                - transition_audio: L'audio de la transition
                - cut_info: Dict avec les points de coupe décidés par l'IA
        """
        print(f"  🤖 L'IA analyse et décide de tout...")
        
        # L'IA prédit tous les paramètres
        p = self._get_ai_params(audio1, audio2)
        
        # === POINTS DE COUPE DÉCIDÉS PAR L'IA ===
        duration1 = len(audio1) / self.sample_rate
        duration2 = len(audio2) / self.sample_rate
        
        # Point de coupe track 1 (où couper)
        raw_cut1 = duration1 * p['track1_cut_position']
        
        # Ajuster sur un beat/mesure si l'IA le décide
        if p['track1_cut_on_bar'] > 0.6:
            cut_time1 = self._find_nearest_bar(audio1, raw_cut1)
        elif p['track1_cut_on_beat'] > 0.6:
            cut_time1 = self._find_nearest_beat(audio1, raw_cut1)
        else:
            cut_time1 = raw_cut1
        
        # Point de départ track 2 (où commencer)
        raw_start2 = duration2 * p['track2_start_position'] * 0.3  # Max 30% dans le morceau
        
        if p['track2_start_on_bar'] > 0.6:
            start_time2 = self._find_nearest_bar(audio2, raw_start2)
        elif p['track2_start_on_beat'] > 0.6:
            start_time2 = self._find_nearest_beat(audio2, raw_start2)
        else:
            start_time2 = raw_start2
        
        # Durée totale ajustée par l'IA
        duration = duration * p['total_duration_factor']
        duration = max(15, min(duration, 35))
        
        print(f"\n  📊 DÉCISIONS DE L'IA:")
        print(f"     ✂️ COUPE Track 1 à: {cut_time1:.2f}s / {duration1:.2f}s")
        print(f"        (sur {'mesure' if p['track1_cut_on_bar'] > 0.6 else 'beat' if p['track1_cut_on_beat'] > 0.6 else 'temps libre'})")
        print(f"     ▶️ DÉBUT Track 2 à: {start_time2:.2f}s")
        print(f"        (sur {'mesure' if p['track2_start_on_bar'] > 0.6 else 'beat' if p['track2_start_on_beat'] > 0.6 else 'temps libre'})")
        print(f"     ⏱️ Durée transition: {duration:.1f}s")
        print(f"     📐 Structure: P1={p['phase1_duration']:.0%} | P2={p['phase2_duration']:.0%} | P3={p['phase3_duration']:.0%}")
        print(f"     🎚️ Filtres: {p['p1_filter_start']*10:.0f}k→{p['p1_filter_end']*10:.0f}k | {p['p3_filter_start']*10:.0f}k→{p['p3_filter_end']*10:.0f}k Hz")
        print(f"     🎛️ Effets: reverb={p['reverb_amount']:.2f}, echo={p['echo_amount']:.2f}")
        print(f"     🎨 Style: smooth={p['style_smooth']:.2f}, dramatic={p['style_dramatic']:.2f}")
        
        # Convertir en samples
        cut_sample1 = int(cut_time1 * self.sample_rate)
        start_sample2 = int(start_time2 * self.sample_rate)
        n_samples = int(duration * self.sample_rate)
        
        # Extraire les segments
        seg1 = audio1[max(0, cut_sample1 - n_samples):cut_sample1]
        if len(seg1) < n_samples:
            seg1 = np.pad(seg1, (n_samples - len(seg1), 0))
        
        seg2 = audio2[start_sample2:start_sample2 + n_samples]
        if len(seg2) < n_samples:
            seg2 = np.pad(seg2, (0, n_samples - len(seg2)))
        
        # Séparer les sources
        print("  🔀 Séparation des sources...")
        comp1 = self.separator.full_separation(seg1)
        comp2 = self.separator.full_separation(seg2)
        
        # Durées des phases
        p1_len = int(n_samples * p['phase1_duration'])
        p2_len = int(n_samples * p['phase2_duration'])
        p3_len = n_samples - p1_len - p2_len
        
        # Générer les phases
        print("  🎭 Construction des phases...")
        phase1 = self._phase1(comp1, p1_len, p)
        phase2 = self._phase2(comp1, comp2, p2_len, p, p1_len)
        phase3 = self._phase3(comp2, p3_len, p, p1_len + p2_len)
        
        # Assembler
        transition = np.concatenate([phase1, phase2, phase3])
        
        # Normaliser
        peak = np.max(np.abs(transition))
        if peak > 0:
            transition = transition * (0.95 / peak)
        
        print(f"  ✓ Transition: {len(transition) / self.sample_rate:.2f}s")
        
        # Retourner la transition et les infos de coupe
        cut_info = {
            'track1_cut_time': cut_time1,
            'track2_start_time': start_time2,
            'transition_duration': duration,
            'params': p
        }
        
        return transition, cut_info
    
    def _phase1(self, comp1, length, p):
        t = np.linspace(0, 1, length)
        
        drums = self._resize(comp1['percussive'], length)
        harm = self._resize(comp1['harmonic'], length)
        bass = self._resize(comp1['bass'], length)
        
        f_start = 500 + p['p1_filter_start'] * 9500
        f_end = 200 + p['p1_filter_end'] * 2000
        harm_filtered = self._filter_sweep(harm, f_start, f_end, 1 + p['p1_filter_curve'])
        
        drums_curve = 1.0 - t * (0.3 + p['style_dramatic'] * 0.3)
        bass_curve = 1.0 - t * 0.4
        
        mix = (drums * drums_curve * p['drums_volume'] +
               harm_filtered * p['harmonic_volume'] +
               bass * bass_curve * p['bass_volume'])
        
        if p['reverb_amount'] > 0.1:
            mix_wet = self._apply_reverb(mix, p['reverb_decay'])
            reverb_curve = t * p['reverb_amount']
            mix = mix * (1 - reverb_curve) + mix_wet * reverb_curve
        
        return mix
    
    def _phase2(self, comp1, comp2, length, p, offset):
        t = np.linspace(0, 1, length)
        
        drums1 = self._resize(comp1['percussive'][offset:], length)
        harm1 = self._resize(comp1['harmonic'][offset:], length)
        bass1 = self._resize(comp1['bass'][offset:], length)
        
        drums2 = self._resize(comp2['percussive'], length)
        harm2 = self._resize(comp2['harmonic'], length)
        bass2 = self._resize(comp2['bass'], length)
        
        curve_type = p['crossfade_curve']
        if curve_type < 0.33:
            curve_out, curve_in = 1 - t, t
        elif curve_type < 0.66:
            curve_out = np.cos(t * np.pi / 2)
            curve_in = np.sin(t * np.pi / 2)
        else:
            curve_out = 1 - t ** 2
            curve_in = t ** 2
        
        harm1_f = self._filter_sweep(harm1, 400, 250, 1.0)
        if p['echo_amount'] > 0.15:
            delay_ms = 200 + p['echo_delay'] * 300
            harm1_f = self._apply_echo(harm1_f, delay_ms, p['echo_feedback'], p['echo_amount'])
        
        track1 = (drums1 * 0.5 + harm1_f + bass1 * 0.3) * curve_out
        
        f_start = 200 + p['p3_filter_start'] * 800
        f_mid = 1000 + p['p3_filter_start'] * 2000
        harm2_f = self._filter_sweep(harm2, f_start, f_mid, 1.0)
        
        bass2_curve = np.where(t > p['bass_swap_point'], 
                               (t - p['bass_swap_point']) / (1 - p['bass_swap_point']), 0)
        track2 = (drums2 * 0.6 + harm2_f + bass2 * bass2_curve * 0.5) * curve_in
        
        mix = track1 * p['harmonic_volume'] + track2 * p['harmonic_volume']
        
        if p['use_riser'] > 0.5:
            riser = self._create_riser(length / self.sample_rate, p['tension_level'] * 0.12)
            riser = self._resize(riser, length)
            mix = mix + riser
        
        return mix
    
    def _phase3(self, comp2, length, p, offset):
        t = np.linspace(0, 1, length)
        
        drums = self._resize(comp2['percussive'][offset:], length)
        harm = self._resize(comp2['harmonic'][offset:], length)
        bass = self._resize(comp2['bass'][offset:], length)
        
        f_start = 1000 + p['p3_filter_start'] * 2000
        f_end = 4000 + p['p3_filter_end'] * 6000
        harm_filtered = self._filter_sweep(harm, f_start, f_end, 1 + p['p3_filter_curve'])
        
        vol_curve = 0.6 + 0.4 * (t ** (0.5 + p['energy_curve'] * 0.5))
        
        mix = (drums * p['drums_volume'] +
               harm_filtered * p['harmonic_volume'] +
               bass * (0.5 + 0.5 * t) * p['bass_volume']) * vol_curve
        
        if p['reverb_amount'] > 0.1:
            mix_wet = self._apply_reverb(mix, p['reverb_decay'] * 0.5)
            reverb_curve = (1 - t) * p['reverb_amount'] * 0.5
            mix = mix * (1 - reverb_curve) + mix_wet * reverb_curve
        
        return mix