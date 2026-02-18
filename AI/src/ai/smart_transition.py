import numpy as np
import torch
import librosa
import os
from scipy.signal import butter, filtfilt, sosfilt

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
    def __init__(self, sample_rate=SAMPLE_RATE, model_path="models/params_vae.pth"):
        self.sample_rate = sample_rate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.separator = AudioSeparator(sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate)
        self.key_analyzer = KeyAnalyzer(sample_rate)
        self.circle_of_fifths = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']
        self.model = None
        self.model_loaded = False
        if MODEL_AVAILABLE and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model = TransitionParamsVAE(
                input_dim=checkpoint.get('input_dim', 28),
                hidden_dim=512,
                latent_dim=checkpoint.get('latent_dim', 128),
                output_dim=checkpoint.get('output_dim', 24)
            ).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model_loaded = True
            print("  Model IA charge")
        except Exception as e:
            print(f"  Erreur model: {e}")
            self.model = None
            self.model_loaded = False

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
        
        key = key_info['key']
        key_pos = self.circle_of_fifths.index(key) / 12.0 if key in self.circle_of_fifths else 0.5
        
        return np.array([
            features['bpm'] / 200.0,
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
        ], dtype=np.float32)

    def _get_ai_params(self, audio1, audio2):
        f1 = self._extract_features(audio1)
        f2 = self._extract_features(audio2)
        
        if self.model is not None and self.model_loaded:
            try:
                input_features = np.concatenate([f1, f2])
                input_tensor = torch.FloatTensor(input_features).unsqueeze(0).to(self.device)
                params_tensor = self.model.predict(input_tensor)
                return self.model.get_params_dict(params_tensor)
            except:
                pass
        return self._default_params()

    def _default_params(self):
        return {
            'low_eq_1': -0.5, 'mid_eq_1': -0.3, 'high_eq_1': -0.2,
            'low_eq_2': -0.8, 'mid_eq_2': -0.2, 'high_eq_2': -0.3,
            'volume_curve_1': 0.6, 'volume_curve_2': 0.6,
            'crossfade_type': 1, 'crossfade_position': 0.5,
            'cue_out_position': 0.8, 'cue_in_position': 0.05,
            'align_to_beat': 0.8, 'align_to_bar': 0.6,
            'transition_beats': 32, 'eq_swap_timing': 0.5, 'bass_swap_beat': 0.5,
            'mix_style': 0, 'filter_sweep': 0.5, 'filter_resonance': 0.3, 'tension_effect': 0,
            'duck_vocals_1': 0.0, 'duck_vocals_2': 0.0, 'energy_direction': 0.5
        }

    def _apply_eq(self, audio, low_db, mid_db, high_db):
        nyq = self.sample_rate / 2
        low_gain = 10 ** (low_db / 20)
        mid_gain = 10 ** (mid_db / 20)
        high_gain = 10 ** (high_db / 20)
        
        try:
            b_low, a_low = butter(2, 200 / nyq, btype='low')
            b_mid, a_mid = butter(2, [200 / nyq, 2000 / nyq], btype='band')
            b_high, a_high = butter(2, 2000 / nyq, btype='high')
            
            low = filtfilt(b_low, a_low, audio) * low_gain
            mid = filtfilt(b_mid, a_mid, audio) * mid_gain
            high = filtfilt(b_high, a_high, audio) * high_gain
            
            return low + mid + high
        except:
            return audio

    def _apply_filter_sweep(self, audio, start_freq, end_freq, resonance=0.3):
        n = len(audio)
        output = np.zeros(n)
        n_seg = 50
        seg_len = n // n_seg
        
        for i in range(n_seg):
            s, e = i * seg_len, min((i + 1) * seg_len, n)
            t = i / (n_seg - 1)
            freq = start_freq + (end_freq - start_freq) * t
            freq = max(100, min(freq, self.sample_rate / 2 - 100))
            
            try:
                b, a = butter(2, freq / (self.sample_rate / 2), btype='low')
                output[s:e] = filtfilt(b, a, audio[s:e])
            except:
                output[s:e] = audio[s:e]
        
        return output

    def _apply_reverb(self, audio, amount):
        if amount < 0.1:
            return audio
        delay = int(0.03 * self.sample_rate)
        output = audio.copy()
        decay = 0.3 * amount
        for i in range(1, 5):
            d = delay * i
            if d < len(audio):
                output[d:] += audio[:-d] * (decay ** i)
        peak = np.max(np.abs(output))
        return output / peak * 0.95 if peak > 1 else output

    def _apply_delay(self, audio, amount, bpm=120):
        if amount < 0.1:
            return audio
        delay = int((60 / bpm / 4) * self.sample_rate)
        output = audio.copy()
        feedback = 0.4 * amount
        for i in range(4):
            d = delay * (i + 1)
            if d < len(audio):
                output[d:] += audio[:-d] * (feedback ** (i + 1))
        peak = np.max(np.abs(output))
        return output / peak * 0.95 if peak > 1 else output

    def _resize(self, arr, length):
        if len(arr) >= length:
            return arr[:length]
        return np.pad(arr, (0, length - len(arr)))

    def _find_beat(self, audio, target_time):
        try:
            _, beat_frames = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
            if len(beat_times) == 0:
                return target_time
            idx = np.argmin(np.abs(beat_times - target_time))
            return beat_times[idx]
        except:
            return target_time

    def _find_bar(self, audio, target_time):
        try:
            _, beat_frames = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
            if len(beat_times) < 4:
                return target_time
            bar_times = beat_times[::4]
            idx = np.argmin(np.abs(bar_times - target_time))
            return bar_times[idx]
        except:
            return target_time

    def generate_transition(self, audio1, audio2, duration=20.0, style='auto'):
        print("  Analyse IA...")
        p = self._get_ai_params(audio1, audio2)
        
        dur1 = len(audio1) / self.sample_rate
        dur2 = len(audio2) / self.sample_rate
        
        raw_cut = dur1 * p['cue_out_position']
        if p['align_to_bar'] > 0.6:
            cut_time = self._find_bar(audio1, raw_cut)
        elif p['align_to_beat'] > 0.6:
            cut_time = self._find_beat(audio1, raw_cut)
        else:
            cut_time = raw_cut
        
        raw_start = dur2 * p['cue_in_position']
        if p['align_to_bar'] > 0.6:
            start_time = self._find_bar(audio2, raw_start)
        elif p['align_to_beat'] > 0.6:
            start_time = self._find_beat(audio2, raw_start)
        else:
            start_time = raw_start
        
        trans_dur = max(15, min(duration * (p['transition_beats'] / 32), 35))
        n_samples = int(trans_dur * self.sample_rate)
        
        mix_styles = ['blend', 'cut', 'filter', 'echo']
        style_name = mix_styles[p['mix_style']] if p['mix_style'] < 4 else 'blend'
        
        tension_types = ['none', 'delay', 'reverb', 'buildup']
        tension_name = tension_types[p['tension_effect']] if p['tension_effect'] < 4 else 'none'
        
        print(f"  Cue out: {cut_time:.1f}s | Cue in: {start_time:.1f}s")
        print(f"  Duree: {trans_dur:.1f}s | Style: {style_name}")
        print(f"  EQ1: L{p['low_eq_1']:+.1f} M{p['mid_eq_1']:+.1f} H{p['high_eq_1']:+.1f}")
        print(f"  EQ2: L{p['low_eq_2']:+.1f} M{p['mid_eq_2']:+.1f} H{p['high_eq_2']:+.1f}")
        
        cut_sample = int(cut_time * self.sample_rate)
        start_sample = int(start_time * self.sample_rate)
        
        seg1 = audio1[max(0, cut_sample - n_samples):cut_sample]
        if len(seg1) < n_samples:
            seg1 = np.pad(seg1, (n_samples - len(seg1), 0))
        
        seg2 = audio2[start_sample:start_sample + n_samples]
        if len(seg2) < n_samples:
            seg2 = np.pad(seg2, (0, n_samples - len(seg2)))
        
        print("  Separation...")
        comp1 = self.separator.full_separation(seg1)
        comp2 = self.separator.full_separation(seg2)
        
        print("  Mix...")
        transition = self._mix_tracks(comp1, comp2, n_samples, p, style_name, tension_name)
        
        peak = np.max(np.abs(transition))
        if peak > 0:
            transition = transition * (0.95 / peak)
        
        print(f"  Transition: {len(transition) / self.sample_rate:.1f}s")
        
        cut_info = {
            'track1_cut_time': cut_time,
            'track2_start_time': start_time,
            'transition_duration': trans_dur,
            'params': p
        }
        
        return transition, cut_info

    def _mix_tracks(self, comp1, comp2, length, p, style, tension):
        t = np.linspace(0, 1, length)
        
        cf_type = p['crossfade_type']
        if cf_type == 0:
            curve_out, curve_in = 1 - t, t
        elif cf_type == 1:
            curve_out = np.cos(t * np.pi / 2)
            curve_in = np.sin(t * np.pi / 2)
        else:
            curve_out = np.sqrt(1 - t)
            curve_in = np.sqrt(t)
        
        low1 = self._resize(comp1['bass'], length)
        mid1 = self._resize(comp1['mids'], length)
        high1 = self._resize(comp1['highs'], length)
        perc1 = self._resize(comp1['percussive'], length)
        harm1 = self._resize(comp1['harmonic'], length)
        
        low2 = self._resize(comp2['bass'], length)
        mid2 = self._resize(comp2['mids'], length)
        high2 = self._resize(comp2['highs'], length)
        perc2 = self._resize(comp2['percussive'], length)
        harm2 = self._resize(comp2['harmonic'], length)
        
        eq_swap = int(length * p['eq_swap_timing'])
        bass_swap = int(length * p['bass_swap_beat'])
        
        low_eq1 = np.ones(length)
        low_eq1[eq_swap:] = np.linspace(1, 10 ** (p['low_eq_1'] / 20), length - eq_swap)
        
        low_eq2 = np.ones(length) * 10 ** (p['low_eq_2'] / 20)
        low_eq2[bass_swap:] = np.linspace(10 ** (p['low_eq_2'] / 20), 1, length - bass_swap)
        
        mid_eq1 = np.linspace(1, 10 ** (p['mid_eq_1'] / 20), length)
        mid_eq2 = np.linspace(10 ** (p['mid_eq_2'] / 20), 1, length)
        
        high_eq1 = np.linspace(1, 10 ** (p['high_eq_1'] / 20), length)
        high_eq2 = np.linspace(10 ** (p['high_eq_2'] / 20), 1, length)
        
        track1_low = low1 * low_eq1 * curve_out
        track1_mid = mid1 * mid_eq1 * curve_out
        track1_high = high1 * high_eq1 * curve_out
        
        track2_low = low2 * low_eq2 * curve_in
        track2_mid = mid2 * mid_eq2 * curve_in
        track2_high = high2 * high_eq2 * curve_in
        
        if style == 'filter' and p['filter_sweep'] > 0.3:
            track1_harm = harm1 * curve_out
            track1_harm = self._apply_filter_sweep(track1_harm, 8000, 300 + p['filter_resonance'] * 500)
            track2_harm = harm2 * curve_in
            track2_harm = self._apply_filter_sweep(track2_harm, 300, 8000)
            mix = track1_low + track1_mid + track1_high + track1_harm * 0.3
            mix += track2_low + track2_mid + track2_high + track2_harm * 0.3
        else:
            mix = track1_low + track1_mid + track1_high
            mix += track2_low + track2_mid + track2_high
        
        if tension == 'delay':
            mix_mid = mid1 * curve_out + mid2 * curve_in
            delayed = self._apply_delay(mix_mid, 0.4)
            mix = mix * 0.85 + delayed * 0.15
        elif tension == 'reverb':
            reverbed = self._apply_reverb(mix, 0.4)
            rev_curve = np.sin(t * np.pi)
            mix = mix * (1 - rev_curve * 0.3) + reverbed * (rev_curve * 0.3)
        elif tension == 'buildup':
            noise = np.random.randn(length) * 0.02
            noise_filtered = self._apply_filter_sweep(noise, 200, 4000)
            buildup_curve = t ** 2
            mix = mix + noise_filtered * buildup_curve * 0.1
        
        if p['duck_vocals_1'] > 0.3:
            duck_amount = p['duck_vocals_1']
            mid_point = length // 2
            duck_curve = np.ones(length)
            duck_curve[:mid_point] = np.linspace(1, 1 - duck_amount, mid_point)
            track1_mid = track1_mid * duck_curve
        
        vol1 = np.ones(length)
        vol2 = np.ones(length)
        
        if p['volume_curve_1'] > 0.5:
            vol1 = 1 - t * (p['volume_curve_1'] - 0.5) * 2
        if p['volume_curve_2'] > 0.5:
            vol2 = t * (p['volume_curve_2'] - 0.5) * 2 + (1 - (p['volume_curve_2'] - 0.5) * 2)
        
        return mix
