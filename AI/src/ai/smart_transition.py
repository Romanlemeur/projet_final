import numpy as np
import torch
import librosa
import os
from scipy.signal import butter, filtfilt, medfilt
from scipy.ndimage import uniform_filter1d

from src.ai.audio_separator import AudioSeparator
from src.analysis.feature_extractor import FeatureExtractor
from src.analysis.key_analyzer import KeyAnalyzer
from src.utils.config import SAMPLE_RATE

try:
    from src.ai.transition_params_model import TransitionParamsVAE
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False


CAMELOT_WHEEL = {
    'C': ('8B', 'major'), 'Am': ('8A', 'minor'),
    'G': ('9B', 'major'), 'Em': ('9A', 'minor'),
    'D': ('10B', 'major'), 'Bm': ('10A', 'minor'),
    'A': ('11B', 'major'), 'F#m': ('11A', 'minor'),
    'E': ('12B', 'major'), 'C#m': ('12A', 'minor'),
    'B': ('1B', 'major'), 'G#m': ('1A', 'minor'),
    'F#': ('2B', 'major'), 'Gb': ('2B', 'major'), 'D#m': ('2A', 'minor'), 'Ebm': ('2A', 'minor'),
    'Db': ('3B', 'major'), 'C#': ('3B', 'major'), 'Bbm': ('3A', 'minor'), 'A#m': ('3A', 'minor'),
    'Ab': ('4B', 'major'), 'G#': ('4B', 'major'), 'Fm': ('4A', 'minor'),
    'Eb': ('5B', 'major'), 'D#': ('5B', 'major'), 'Cm': ('5A', 'minor'),
    'Bb': ('6B', 'major'), 'A#': ('6B', 'major'), 'Gm': ('6A', 'minor'),
    'F': ('7B', 'major'), 'Dm': ('7A', 'minor')
}

CAMELOT_COMPATIBLE = {
    '1A': ['1A', '1B', '12A', '2A'], '1B': ['1B', '1A', '12B', '2B'],
    '2A': ['2A', '2B', '1A', '3A'], '2B': ['2B', '2A', '1B', '3B'],
    '3A': ['3A', '3B', '2A', '4A'], '3B': ['3B', '3A', '2B', '4B'],
    '4A': ['4A', '4B', '3A', '5A'], '4B': ['4B', '4A', '3B', '5B'],
    '5A': ['5A', '5B', '4A', '6A'], '5B': ['5B', '5A', '4B', '6B'],
    '6A': ['6A', '6B', '5A', '7A'], '6B': ['6B', '6A', '5B', '7B'],
    '7A': ['7A', '7B', '6A', '8A'], '7B': ['7B', '7A', '6B', '8B'],
    '8A': ['8A', '8B', '7A', '9A'], '8B': ['8B', '8A', '7B', '9B'],
    '9A': ['9A', '9B', '8A', '10A'], '9B': ['9B', '9A', '8B', '10B'],
    '10A': ['10A', '10B', '9A', '11A'], '10B': ['10B', '10A', '9B', '11B'],
    '11A': ['11A', '11B', '10A', '12A'], '11B': ['11B', '11A', '10B', '12B'],
    '12A': ['12A', '12B', '11A', '1A'], '12B': ['12B', '12A', '11B', '1B']
}


class VocalDetector:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.frame_length = 2048

    def detect_vocals_curve(self, audio):
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13, hop_length=self.hop_length)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=audio, hop_length=self.hop_length)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate, hop_length=self.hop_length)
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
        
        mfcc_var = np.var(mfcc[1:6], axis=0)
        mfcc_energy = np.mean(np.abs(mfcc[1:5]), axis=0)
        centroid_norm = spectral_centroid / (self.sample_rate / 2)
        contrast_mid = np.mean(spectral_contrast[2:5], axis=0)
        
        vocal_score = np.zeros(len(rms))
        vocal_score += (centroid_norm > 0.03) * (centroid_norm < 0.28) * 0.22
        vocal_score += (spectral_flatness < 0.18) * 0.16
        vocal_score += (mfcc_var > np.percentile(mfcc_var, 22)) * 0.20
        vocal_score += (mfcc_energy > np.percentile(mfcc_energy, 25)) * 0.14
        vocal_score += (zcr > 0.012) * (zcr < 0.09) * 0.10
        vocal_score += (rms > np.percentile(rms, 12)) * 0.08
        vocal_score += (contrast_mid > np.percentile(contrast_mid, 28)) * 0.10
        
        vocal_score = medfilt(vocal_score, kernel_size=17)
        vocal_score = uniform_filter1d(vocal_score, size=13)
        return np.clip(vocal_score, 0, 1)

    def get_vocal_segments(self, audio, threshold=0.32):
        scores = self.detect_vocals_curve(audio)
        is_vocal = scores > threshold
        
        segments = []
        in_segment = False
        start = 0
        
        for i, v in enumerate(is_vocal):
            if v and not in_segment:
                start = i
                in_segment = True
            elif not v and in_segment:
                t_start = start * self.hop_length / self.sample_rate
                t_end = i * self.hop_length / self.sample_rate
                if t_end - t_start > 0.5:
                    segments.append({
                        'start': t_start,
                        'end': t_end,
                        'duration': t_end - t_start,
                        'avg_intensity': np.mean(scores[start:i])
                    })
                in_segment = False
        
        return segments

    def get_instrumental_segments(self, audio, min_duration=2.5, threshold=0.28):
        scores = self.detect_vocals_curve(audio)
        is_instrumental = scores < threshold
        
        segments = []
        in_segment = False
        start = 0
        
        for i, inst in enumerate(is_instrumental):
            if inst and not in_segment:
                start = i
                in_segment = True
            elif not inst and in_segment:
                t_start = start * self.hop_length / self.sample_rate
                t_end = i * self.hop_length / self.sample_rate
                if t_end - t_start >= min_duration:
                    segments.append({
                        'start': t_start,
                        'end': t_end,
                        'duration': t_end - t_start,
                        'vocal_score': np.mean(scores[start:i])
                    })
                in_segment = False
        
        if in_segment:
            t_start = start * self.hop_length / self.sample_rate
            t_end = len(is_instrumental) * self.hop_length / self.sample_rate
            if t_end - t_start >= min_duration:
                segments.append({
                    'start': t_start,
                    'end': t_end,
                    'duration': t_end - t_start,
                    'vocal_score': np.mean(scores[start:])
                })
        
        return sorted(segments, key=lambda x: x['vocal_score'])

    def has_vocals_in_range(self, audio, start_time, end_time, threshold=0.32):
        scores = self.detect_vocals_curve(audio)
        start_frame = max(0, int(start_time * self.sample_rate / self.hop_length))
        end_frame = min(len(scores), int(end_time * self.sample_rate / self.hop_length))
        
        if end_frame > start_frame:
            segment_scores = scores[start_frame:end_frame]
            return np.mean(segment_scores > threshold) > 0.22
        return False

    def get_vocal_intensity(self, audio, time, window=1.5):
        scores = self.detect_vocals_curve(audio)
        center = int(time * self.sample_rate / self.hop_length)
        half_win = int(window * self.sample_rate / self.hop_length / 2)
        start = max(0, center - half_win)
        end = min(len(scores), center + half_win)
        return np.mean(scores[start:end]) if end > start else 0.5


class StructureAnalyzer:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.hop_length = 512

    def get_energy_curve(self, audio):
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        return uniform_filter1d(rms, size=25)

    def get_percussive_energy(self, audio):
        try:
            _, percussive = librosa.effects.hpss(audio)
            return librosa.feature.rms(y=percussive, hop_length=self.hop_length)[0]
        except:
            return self.get_energy_curve(audio)

    def get_spectral_flux(self, audio):
        stft = np.abs(librosa.stft(audio, hop_length=self.hop_length))
        flux = np.sum(np.diff(stft, axis=1) ** 2, axis=0)
        return uniform_filter1d(flux, size=15)

    def detect_sections(self, audio):
        duration = len(audio) / self.sample_rate
        energy = self.get_energy_curve(audio)
        perc = self.get_percussive_energy(audio)
        flux = self.get_spectral_flux(audio)
        
        frame_dur = self.hop_length / self.sample_rate
        
        e_high = np.percentile(energy, 72)
        e_low = np.percentile(energy, 32)
        p_high = np.percentile(perc, 65)
        p_low = np.percentile(perc, 35)
        
        sections = []
        
        intro_end = self._find_first_energy_rise(energy, e_low, frame_dur)
        if intro_end > 2:
            sections.append({
                'type': 'intro',
                'start': 0,
                'end': intro_end,
                'energy': 'low',
                'loopable': True
            })
        
        outro_start = self._find_last_energy_drop(energy, e_low, frame_dur, duration)
        if outro_start < duration - 5:
            sections.append({
                'type': 'outro',
                'start': outro_start,
                'end': duration,
                'energy': 'low',
                'loopable': True
            })
        
        breakdowns = self._find_breakdowns(energy, perc, e_low, p_low, frame_dur, duration)
        for bd in breakdowns:
            sections.append({
                'type': 'breakdown',
                'start': bd['start'],
                'end': bd['end'],
                'energy': 'low',
                'loopable': True
            })
        
        drops = self._find_drops(energy, perc, e_high, p_high, frame_dur)
        for drop in drops:
            sections.append({
                'type': 'drop',
                'start': drop['start'],
                'end': drop['end'],
                'energy': 'high',
                'loopable': False
            })
        
        return sorted(sections, key=lambda x: x['start'])

    def _find_first_energy_rise(self, energy, threshold, frame_dur):
        for i, e in enumerate(energy):
            if e > threshold:
                return max(0, i * frame_dur - 1)
        return 10

    def _find_last_energy_drop(self, energy, threshold, frame_dur, duration):
        for i in range(len(energy) - 1, -1, -1):
            if energy[i] > threshold:
                return min(duration, (i + 1) * frame_dur)
        return duration - 15

    def _find_breakdowns(self, energy, perc, e_thresh, p_thresh, frame_dur, duration):
        min_len = min(len(energy), len(perc))
        is_breakdown = (energy[:min_len] < e_thresh) & (perc[:min_len] < p_thresh)
        is_breakdown = medfilt(is_breakdown.astype(float), kernel_size=35) > 0.5
        
        breakdowns = []
        in_bd = False
        start = 0
        
        for i, bd in enumerate(is_breakdown):
            t = i * frame_dur
            if t < duration * 0.12 or t > duration * 0.88:
                continue
            if bd and not in_bd:
                start = i
                in_bd = True
            elif not bd and in_bd:
                t_start = start * frame_dur
                t_end = i * frame_dur
                if t_end - t_start > 2.5:
                    breakdowns.append({
                        'start': t_start,
                        'end': t_end,
                        'duration': t_end - t_start,
                        'energy': np.mean(energy[start:i])
                    })
                in_bd = False
        
        return sorted(breakdowns, key=lambda x: x['energy'])

    def _find_drops(self, energy, perc, e_thresh, p_thresh, frame_dur):
        min_len = min(len(energy), len(perc))
        is_drop = (energy[:min_len] > e_thresh) & (perc[:min_len] > p_thresh)
        is_drop = medfilt(is_drop.astype(float), kernel_size=21) > 0.5
        
        drops = []
        in_drop = False
        start = 0
        
        for i, d in enumerate(is_drop):
            if d and not in_drop:
                start = i
                in_drop = True
            elif not d and in_drop:
                t_start = start * frame_dur
                t_end = i * frame_dur
                if t_end - t_start > 3:
                    drops.append({'start': t_start, 'end': t_end})
                in_drop = False
        
        return drops

    def get_section_stability(self, audio, start_time, end_time):
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        segment = audio[start_sample:end_sample]
        
        if len(segment) < self.sample_rate:
            return 0
        
        rms = librosa.feature.rms(y=segment, hop_length=self.hop_length)[0]
        if len(rms) < 2:
            return 0
        
        rms_stability = 1.0 - min(np.std(rms) / (np.mean(rms) + 1e-8), 1.0)
        
        try:
            tempo_curve = librosa.feature.tempogram(y=segment, sr=self.sample_rate)
            tempo_stability = 1.0 - min(np.std(np.max(tempo_curve, axis=0)) / 10, 1.0)
        except:
            tempo_stability = 0.5
        
        return rms_stability * 0.6 + tempo_stability * 0.4


class PhraseDetector:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate

    def detect_beats_and_phrases(self, audio):
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
        
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])
        
        beat_duration = 60.0 / tempo if tempo > 0 else 0.5
        bar_duration = beat_duration * 4
        
        bars = beat_times[::4] if len(beat_times) >= 4 else beat_times
        phrases_4bar = beat_times[::16] if len(beat_times) >= 16 else bars
        phrases_8bar = beat_times[::32] if len(beat_times) >= 32 else phrases_4bar
        phrases_16bar = beat_times[::64] if len(beat_times) >= 64 else phrases_8bar
        phrases_32bar = beat_times[::128] if len(beat_times) >= 128 else phrases_16bar
        
        try:
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            pulse = librosa.beat.plp(onset_envelope=onset_env, sr=self.sample_rate)
            time_signature = 4 if np.argmax(np.mean(pulse, axis=1)) % 4 == 0 else 3
        except:
            time_signature = 4
        
        return {
            'tempo': tempo,
            'time_signature': time_signature,
            'beats': beat_times,
            'beat_duration': beat_duration,
            'bar_duration': bar_duration,
            'bars': bars,
            'phrases_4bar': phrases_4bar,
            'phrases_8bar': phrases_8bar,
            'phrases_16bar': phrases_16bar,
            'phrases_32bar': phrases_32bar
        }

    def snap_to_bar(self, bars, target_time, direction='nearest'):
        if len(bars) == 0:
            return target_time
        
        if direction == 'nearest':
            idx = np.argmin(np.abs(bars - target_time))
        elif direction == 'before':
            valid = bars[bars <= target_time]
            idx = len(valid) - 1 if len(valid) > 0 else 0
        else:
            valid = bars[bars >= target_time]
            idx = np.where(bars >= target_time)[0][0] if len(valid) > 0 else len(bars) - 1
        
        return bars[idx]

    def snap_to_phrase(self, phrases, target_time, direction='nearest'):
        return self.snap_to_bar(phrases, target_time, direction)

    def get_phrase_at_time(self, phrases, time):
        if len(phrases) < 2:
            return None
        
        for i in range(len(phrases) - 1):
            if phrases[i] <= time < phrases[i + 1]:
                return {'start': phrases[i], 'end': phrases[i + 1], 'index': i}
        
        return {'start': phrases[-1], 'end': phrases[-1] + (phrases[-1] - phrases[-2]), 'index': len(phrases) - 1}


class HarmonicMixer:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.key_analyzer = KeyAnalyzer(sample_rate)

    def get_camelot(self, key, mode):
        key_clean = key.replace(' ', '')
        
        if mode == 'minor' and not key_clean.endswith('m'):
            key_clean = key_clean + 'm'
        elif mode == 'major' and key_clean.endswith('m'):
            key_clean = key_clean[:-1]
        
        for k, (camelot, m) in CAMELOT_WHEEL.items():
            if k.lower() == key_clean.lower():
                return camelot
        
        for k, (camelot, m) in CAMELOT_WHEEL.items():
            if k.lower().startswith(key.lower()[0]):
                if (mode == 'minor') == (m == 'minor'):
                    return camelot
        
        return '8A' if mode == 'minor' else '8B'

    def check_compatibility(self, key1, mode1, key2, mode2):
        cam1 = self.get_camelot(key1, mode1)
        cam2 = self.get_camelot(key2, mode2)
        
        if cam1 == cam2:
            return 1.0, 'perfect', 'Same key - perfect blend'
        
        if cam2 in CAMELOT_COMPATIBLE.get(cam1, []):
            if cam1[:-1] == cam2[:-1]:
                return 0.95, 'relative', 'Relative major/minor - smooth transition'
            else:
                return 0.88, 'adjacent', 'Adjacent key - energy shift'
        
        num1, num2 = int(cam1[:-1]), int(cam2[:-1])
        dist = min(abs(num1 - num2), 12 - abs(num1 - num2))
        
        if dist == 2:
            return 0.70, 'acceptable', 'Two steps away - use with caution'
        elif dist <= 3:
            return 0.50, 'tension', 'Creates tension - quick transition recommended'
        else:
            return 0.25, 'clash', 'Key clash - avoid long blend'

    def get_energy_boost_key(self, camelot):
        num = int(camelot[:-1])
        letter = camelot[-1]
        new_num = ((num + 6) % 12) + 1
        return f"{new_num}{letter}"


class LoopFinder:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.vocal_detector = VocalDetector(sample_rate)
        self.structure_analyzer = StructureAnalyzer(sample_rate)
        self.phrase_detector = PhraseDetector(sample_rate)

    def score_loop(self, audio, start_time, end_time, beat_info, position='any'):
        score = 0
        reasons = []
        
        has_vocals = self.vocal_detector.has_vocals_in_range(audio, start_time, end_time, threshold=0.28)
        if has_vocals:
            score -= 70
            reasons.append("VOCALS_DETECTED")
        else:
            score += 40
            reasons.append("NO_VOCALS")
        
        stability = self.structure_analyzer.get_section_stability(audio, start_time, end_time)
        score += stability * 25
        reasons.append(f"STABILITY_{stability:.0%}")
        
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        segment = audio[start_sample:end_sample]
        
        if len(segment) > self.sample_rate:
            try:
                _, percussive = librosa.effects.hpss(segment)
                perc_ratio = np.sqrt(np.mean(percussive ** 2)) / (np.sqrt(np.mean(segment ** 2)) + 1e-8)
                
                if perc_ratio < 0.30:
                    score += 22
                    reasons.append("LOW_PERCUSSION")
                elif perc_ratio < 0.45:
                    score += 12
                    reasons.append("MED_PERCUSSION")
            except:
                pass
        
        bars = beat_info['bars']
        if len(bars) > 0:
            start_dist = np.min(np.abs(bars - start_time))
            end_dist = np.min(np.abs(bars - end_time))
            if start_dist < 0.12:
                score += 12
                reasons.append("START_ON_BAR")
            if end_dist < 0.12:
                score += 12
                reasons.append("END_ON_BAR")
        
        phrases = beat_info['phrases_8bar']
        if len(phrases) > 0:
            phrase_dist = np.min(np.abs(phrases - start_time))
            if phrase_dist < 0.25:
                score += 18
                reasons.append("START_ON_PHRASE")
        
        duration = len(audio) / self.sample_rate
        if position == 'outro':
            if start_time > duration * 0.65:
                score += 10
                reasons.append("GOOD_OUTRO_POSITION")
        elif position == 'intro':
            if end_time < duration * 0.35:
                score += 10
                reasons.append("GOOD_INTRO_POSITION")
        
        return score, has_vocals, reasons

    def find_best_loop(self, audio, target_time, n_bars=8, search_range=30.0, position='any'):
        duration = len(audio) / self.sample_rate
        beat_info = self.phrase_detector.detect_beats_and_phrases(audio)
        bar_dur = beat_info['bar_duration']
        loop_dur = bar_dur * n_bars
        bars = beat_info['bars']
        
        if len(bars) == 0:
            return target_time, target_time + loop_dur, True, []
        
        sections = self.structure_analyzer.detect_sections(audio)
        loopable_sections = [s for s in sections if s.get('loopable', False)]
        
        instrumental = self.vocal_detector.get_instrumental_segments(audio, min_duration=loop_dur * 0.8)
        
        priority_zones = []
        for sect in loopable_sections:
            for inst in instrumental[:8]:
                o_start = max(sect['start'], inst['start'])
                o_end = min(sect['end'], inst['end'])
                if o_end - o_start >= loop_dur:
                    priority_zones.append({
                        'start': o_start,
                        'end': o_end,
                        'bonus': 30,
                        'type': sect['type']
                    })
        
        candidates = []
        
        for bar_time in bars:
            if abs(bar_time - target_time) > search_range:
                continue
            if bar_time < 2.5 or bar_time + loop_dur > duration - 2.5:
                continue
            
            loop_end = bar_time + loop_dur
            
            end_bars = bars[bars >= loop_end - 0.15]
            if len(end_bars) > 0:
                loop_end = end_bars[0]
            
            score, has_vocals, reasons = self.score_loop(audio, bar_time, loop_end, beat_info, position)
            
            for pz in priority_zones:
                if bar_time >= pz['start'] - 0.5 and loop_end <= pz['end'] + 0.5:
                    score += pz['bonus']
                    reasons.append(f"IN_{pz['type'].upper()}")
                    break
            
            dist_penalty = abs(bar_time - target_time) / search_range * 10
            score -= dist_penalty
            
            candidates.append({
                'start': bar_time,
                'end': loop_end,
                'score': score,
                'has_vocals': has_vocals,
                'duration': loop_end - bar_time,
                'reasons': reasons
            })
        
        if not candidates:
            return target_time, target_time + loop_dur, True, ["NO_CANDIDATES"]
        
        no_vocal = [c for c in candidates if not c['has_vocals']]
        pool = no_vocal if no_vocal else candidates
        
        best = max(pool, key=lambda x: x['score'])
        return best['start'], best['end'], best['has_vocals'], best['reasons']

    def find_outro_loop(self, audio, n_bars=8):
        duration = len(audio) / self.sample_rate
        target = duration * 0.75
        return self.find_best_loop(audio, target, n_bars, search_range=duration * 0.22, position='outro')

    def find_intro_loop(self, audio, n_bars=8):
        duration = len(audio) / self.sample_rate
        target = min(18.0, duration * 0.15)
        return self.find_best_loop(audio, target, n_bars, search_range=22.0, position='intro')

    def find_breakdown_loop(self, audio, n_bars=8):
        duration = len(audio) / self.sample_rate
        
        energy = self.structure_analyzer.get_energy_curve(audio)
        perc = self.structure_analyzer.get_percussive_energy(audio)
        
        breakdowns = self.structure_analyzer._find_breakdowns(
            energy, perc,
            np.percentile(energy, 32),
            np.percentile(perc, 35),
            self.structure_analyzer.hop_length / self.sample_rate,
            duration
        )
        
        if breakdowns:
            best_bd = breakdowns[0]
            target = (best_bd['start'] + best_bd['end']) / 2
            return self.find_best_loop(audio, target, n_bars, search_range=best_bd['duration'], position='any')
        
        return self.find_best_loop(audio, duration * 0.5, n_bars, position='any')


class SmartTransitionGenerator:
    def __init__(self, sample_rate=SAMPLE_RATE, model_path="models/params_vae.pth"):
        self.sample_rate = sample_rate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.separator = AudioSeparator(sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate)
        self.key_analyzer = KeyAnalyzer(sample_rate)
        self.vocal_detector = VocalDetector(sample_rate)
        self.structure_analyzer = StructureAnalyzer(sample_rate)
        self.phrase_detector = PhraseDetector(sample_rate)
        self.harmonic_mixer = HarmonicMixer(sample_rate)
        self.loop_finder = LoopFinder(sample_rate)
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
            print("  [OK] IA Model loaded")
        except Exception as e:
            print(f"  [!] Model error: {e}")

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

    def analyze_track(self, audio, name="Track"):
        print(f"\n  === ANALYSE: {name} ===")
        
        key_info = self.key_analyzer.detect_key(audio)
        beat_info = self.phrase_detector.detect_beats_and_phrases(audio)
        camelot = self.harmonic_mixer.get_camelot(key_info['key'], key_info['mode'])
        
        sections = self.structure_analyzer.detect_sections(audio)
        vocal_segments = self.vocal_detector.get_vocal_segments(audio)
        instrumental_segments = self.vocal_detector.get_instrumental_segments(audio)
        
        duration = len(audio) / self.sample_rate
        
        print(f"  Duration: {duration:.1f}s")
        print(f"  BPM: {beat_info['tempo']:.1f}")
        print(f"  Time Signature: {beat_info['time_signature']}/4")
        print(f"  Key: {key_info['key']} {key_info['mode']} ({camelot})")
        print(f"  Key Confidence: {key_info['confidence']:.0%}")
        
        print(f"\n  Structure ({len(sections)} sections):")
        for s in sections[:6]:
            print(f"    - {s['type'].upper()}: {s['start']:.1f}s - {s['end']:.1f}s")
        
        print(f"\n  Vocal Segments ({len(vocal_segments)}):")
        for v in vocal_segments[:4]:
            print(f"    - {v['start']:.1f}s - {v['end']:.1f}s ({v['duration']:.1f}s)")
        
        print(f"\n  Instrumental Zones ({len(instrumental_segments)}):")
        for i in instrumental_segments[:4]:
            print(f"    - {i['start']:.1f}s - {i['end']:.1f}s (vocal: {i['vocal_score']:.0%})")
        
        return {
            'duration': duration,
            'key': key_info['key'],
            'mode': key_info['mode'],
            'camelot': camelot,
            'key_confidence': key_info['confidence'],
            'tempo': beat_info['tempo'],
            'time_signature': beat_info['time_signature'],
            'beat_info': beat_info,
            'sections': sections,
            'vocal_segments': vocal_segments,
            'instrumental_segments': instrumental_segments
        }

    def _create_seamless_loop(self, audio, start_time, end_time, target_duration, tempo):
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        loop_segment = audio[start_sample:end_sample]
        
        if len(loop_segment) == 0:
            return np.zeros(int(target_duration * self.sample_rate))
        
        xfade_samples = int(0.025 * self.sample_rate)
        if len(loop_segment) > xfade_samples * 4:
            fade = np.linspace(0, 1, xfade_samples)
            
            end_part = loop_segment[-xfade_samples:].copy()
            start_part = loop_segment[:xfade_samples].copy()
            
            crossfade = end_part * (1 - fade) + start_part * fade
            loop_segment = np.concatenate([
                loop_segment[xfade_samples:-xfade_samples],
                crossfade
            ])
        
        loop_dur = len(loop_segment) / self.sample_rate
        n_repeats = int(np.ceil(target_duration / loop_dur)) + 1
        
        looped = np.tile(loop_segment, n_repeats)
        target_samples = int(target_duration * self.sample_rate)
        
        if len(looped) > target_samples:
            looped = looped[:target_samples]
        elif len(looped) < target_samples:
            looped = np.pad(looped, (0, target_samples - len(looped)))
        
        return looped

    def _apply_filter_sweep(self, audio, start_freq, end_freq):
        n = len(audio)
        output = np.zeros(n)
        n_seg = 55
        seg_len = n // n_seg
        
        for i in range(n_seg):
            s, e = i * seg_len, min((i + 1) * seg_len, n)
            t = i / (n_seg - 1)
            freq = start_freq * ((end_freq / start_freq) ** t)
            freq = max(70, min(freq, self.sample_rate / 2 - 100))
            
            try:
                b, a = butter(3, freq / (self.sample_rate / 2), btype='low')
                output[s:e] = filtfilt(b, a, audio[s:e])
            except:
                output[s:e] = audio[s:e]
        
        return output

    def _apply_reverb(self, audio, amount):
        if amount < 0.05:
            return audio
        delays = [int(d * self.sample_rate / 1000) for d in [17, 29, 41, 53]]
        output = audio.copy()
        for i, delay in enumerate(delays):
            gain = (0.30 ** (i + 1)) * amount
            if delay < len(audio):
                output[delay:] += audio[:-delay] * gain
        peak = np.max(np.abs(output))
        return output / peak * 0.95 if peak > 1 else output

    def _apply_echo(self, audio, amount, tempo):
        if amount < 0.05:
            return audio
        delay = int((60.0 / tempo * 0.5) * self.sample_rate)
        output = audio.copy()
        for i in range(3):
            d = delay * (i + 1)
            if d < len(audio):
                output[d:] += audio[:-d] * ((0.40 * amount) ** (i + 1))
        peak = np.max(np.abs(output))
        return output / peak * 0.95 if peak > 1 else output

    def _resize(self, arr, length):
        if len(arr) >= length:
            return arr[:length]
        return np.pad(arr, (0, length - len(arr)))

    def generate_transition(self, audio1, audio2, duration=20.0, style='auto'):
        print("\n" + "=" * 60)
        print("  DJ TRANSITION GENERATOR - PROFESSIONAL EDITION")
        print("=" * 60)
        
        analysis1 = self.analyze_track(audio1, "TRACK A")
        analysis2 = self.analyze_track(audio2, "TRACK B")
        
        print("\n  === HARMONIC COMPATIBILITY ===")
        harm_score, harm_type, harm_desc = self.harmonic_mixer.check_compatibility(
            analysis1['key'], analysis1['mode'],
            analysis2['key'], analysis2['mode']
        )
        print(f"  {analysis1['camelot']} -> {analysis2['camelot']}")
        print(f"  Compatibility: {harm_type.upper()} ({harm_score:.0%})")
        print(f"  {harm_desc}")
        
        p = self._get_ai_params(audio1, audio2)
        
        tempo1, tempo2 = analysis1['tempo'], analysis2['tempo']
        avg_tempo = (tempo1 + tempo2) / 2
        bpm_diff = abs(tempo1 - tempo2) / avg_tempo * 100
        
        print(f"\n  === TEMPO SYNC ===")
        print(f"  Track A: {tempo1:.1f} BPM")
        print(f"  Track B: {tempo2:.1f} BPM")
        print(f"  Difference: {bpm_diff:.1f}%")
        if bpm_diff > 3:
            print(f"  -> Time-stretch required")
        
        n_bars = 8 if p['transition_beats'] >= 32 else 4
        
        print(f"\n  === LOOP DETECTION ===")
        print(f"  Target: {n_bars} bars")
        
        print(f"\n  Track A - Finding outro loop...")
        loop1_start, loop1_end, has_v1, reasons1 = self.loop_finder.find_outro_loop(audio1, n_bars=n_bars)
        print(f"    Position: {loop1_start:.1f}s - {loop1_end:.1f}s ({loop1_end - loop1_start:.1f}s)")
        print(f"    Vocals: {'YES' if has_v1 else 'NO'}")
        print(f"    Criteria: {', '.join(reasons1[:4])}")
        
        if has_v1:
            print(f"    -> Searching alternative breakdown...")
            loop1_start, loop1_end, has_v1, reasons1 = self.loop_finder.find_breakdown_loop(audio1, n_bars=n_bars)
            print(f"    New: {loop1_start:.1f}s - {loop1_end:.1f}s | Vocals: {'YES' if has_v1 else 'NO'}")
        
        print(f"\n  Track B - Finding intro loop...")
        loop2_start, loop2_end, has_v2, reasons2 = self.loop_finder.find_intro_loop(audio2, n_bars=n_bars)
        print(f"    Position: {loop2_start:.1f}s - {loop2_end:.1f}s ({loop2_end - loop2_start:.1f}s)")
        print(f"    Vocals: {'YES' if has_v2 else 'NO'}")
        print(f"    Criteria: {', '.join(reasons2[:4])}")
        
        if has_v2:
            print(f"    -> Searching alternative breakdown...")
            loop2_start, loop2_end, has_v2, reasons2 = self.loop_finder.find_breakdown_loop(audio2, n_bars=n_bars)
            print(f"    New: {loop2_start:.1f}s - {loop2_end:.1f}s | Vocals: {'YES' if has_v2 else 'NO'}")
        
        trans_beats = p['transition_beats']
        trans_dur = (trans_beats / avg_tempo) * 60
        trans_dur = max(12, min(trans_dur, 48))
        
        print(f"\n  === TRANSITION BUILD ===")
        print(f"  Duration: {trans_dur:.1f}s ({trans_beats:.0f} beats)")
        print(f"  Mix tempo: {avg_tempo:.1f} BPM")
        
        n_samples = int(trans_dur * self.sample_rate)
        
        print(f"\n  Creating seamless loops...")
        loop1_audio = self._create_seamless_loop(audio1, loop1_start, loop1_end, trans_dur * 0.55, tempo1)
        loop2_audio = self._create_seamless_loop(audio2, loop2_start, loop2_end, trans_dur * 0.55, tempo2)
        
        if bpm_diff > 2.5:
            print(f"  Applying time-stretch...")
            rate1 = tempo1 / avg_tempo
            rate2 = tempo2 / avg_tempo
            loop1_audio = librosa.effects.time_stretch(loop1_audio, rate=rate1)
            loop2_audio = librosa.effects.time_stretch(loop2_audio, rate=rate2)
        
        loop1_audio = self._resize(loop1_audio, int(trans_dur * 0.55 * self.sample_rate))
        loop2_audio = self._resize(loop2_audio, int(trans_dur * 0.55 * self.sample_rate))
        
        print(f"  Separating stems...")
        comp1 = self.separator.full_separation(loop1_audio)
        comp2 = self.separator.full_separation(loop2_audio)
        
        print(f"  Mixing with EQ and effects...")
        transition = self._mix_loops(comp1, comp2, n_samples, p, avg_tempo, harm_score)
        
        peak = np.max(np.abs(transition))
        if peak > 0:
            transition = transition * (0.95 / peak)
        
        print(f"\n  === RESULT ===")
        print(f"  Transition: {len(transition) / self.sample_rate:.1f}s")
        print(f"  Harmonic blend: {harm_type}")
        print(f"  Vocal conflicts: {'NONE' if (not has_v1 and not has_v2) else 'MANAGED'}")
        print("=" * 60)
        
        return transition, {
            'track1_cut_time': loop1_start,
            'track2_start_time': loop2_start,
            'transition_duration': trans_dur,
            'loop1': {
                'start': loop1_start,
                'end': loop1_end,
                'has_vocals': has_v1,
                'reasons': reasons1
            },
            'loop2': {
                'start': loop2_start,
                'end': loop2_end,
                'has_vocals': has_v2,
                'reasons': reasons2
            },
            'analysis': {
                'track1': analysis1,
                'track2': analysis2,
                'harmony': {
                    'score': harm_score,
                    'type': harm_type,
                    'description': harm_desc
                }
            },
            'params': p
        }

    def _mix_loops(self, comp1, comp2, length, p, tempo, harmony_score):
        t = np.linspace(0, 1, length)
        
        cf_type = p['crossfade_type']
        if cf_type == 0:
            curve_out = 1 - t
            curve_in = t
        elif cf_type == 1:
            curve_out = np.cos(t * np.pi / 2) ** 2
            curve_in = np.sin(t * np.pi / 2) ** 2
        else:
            curve_out = (1 - t) ** 1.5
            curve_in = t ** 1.5
        
        beat_samples = int(60 / tempo * self.sample_rate)
        bar_samples = beat_samples * 4
        
        low1 = self._resize(comp1['bass'], length)
        mid1 = self._resize(comp1['mids'], length)
        high1 = self._resize(comp1['highs'], length)
        harm1 = self._resize(comp1['harmonic'], length)
        
        low2 = self._resize(comp2['bass'], length)
        mid2 = self._resize(comp2['mids'], length)
        high2 = self._resize(comp2['highs'], length)
        harm2 = self._resize(comp2['harmonic'], length)
        
        bass_swap = int(length * p['bass_swap_beat'])
        bass_swap = (bass_swap // bar_samples) * bar_samples
        bass_swap = max(bar_samples, min(bass_swap, length - bar_samples))
        
        low_mix = np.zeros(length)
        low_mix[:bass_swap] = low1[:bass_swap]
        
        swap_len = min(bar_samples, length - bass_swap)
        if swap_len > 0:
            swap_t = np.linspace(0, 1, swap_len)
            swap_out = np.cos(swap_t * np.pi / 2)
            swap_in = np.sin(swap_t * np.pi / 2)
            low_mix[bass_swap:bass_swap + swap_len] = (
                low1[bass_swap:bass_swap + swap_len] * swap_out +
                low2[bass_swap:bass_swap + swap_len] * swap_in
            )
        
        if bass_swap + swap_len < length:
            low_mix[bass_swap + swap_len:] = low2[bass_swap + swap_len:]
        
        mid_eq1 = np.linspace(1, 10 ** (p['mid_eq_1'] / 20), length)
        mid_eq2 = np.linspace(10 ** (p['mid_eq_2'] / 20), 1, length)
        high_eq1 = np.linspace(1, 10 ** (p['high_eq_1'] / 20), length)
        high_eq2 = np.linspace(10 ** (p['high_eq_2'] / 20), 1, length)
        
        mid_mix = mid1 * mid_eq1 * curve_out + mid2 * mid_eq2 * curve_in
        high_mix = high1 * high_eq1 * curve_out + high2 * high_eq2 * curve_in
        
        harm_blend = max(0.25, harmony_score * 0.45)
        if p['filter_sweep'] > 0.3:
            harm1_f = self._apply_filter_sweep(harm1, 5500, 280)
            harm2_f = self._apply_filter_sweep(harm2, 280, 5500)
            harm_mix = harm1_f * curve_out * harm_blend + harm2_f * curve_in * harm_blend
        else:
            harm_mix = harm1 * curve_out * harm_blend + harm2 * curve_in * harm_blend
        
        mix = low_mix * 0.35 + mid_mix * 0.28 + high_mix * 0.17 + harm_mix * 0.20
        
        tension = p['tension_effect']
        if tension == 1:
            mix = self._apply_echo(mix, 0.22, tempo)
        elif tension == 2:
            rev_curve = np.sin(t * np.pi)
            mix_rev = self._apply_reverb(mix, 0.32)
            mix = mix * (1 - rev_curve * 0.18) + mix_rev * rev_curve * 0.18
        elif tension == 3:
            noise = np.random.randn(length) * 0.004
            noise = self._apply_filter_sweep(noise, 90, 1600)
            mix = mix + noise * (t ** 2.5) * 0.035
        
        return mix
