import numpy as np
import torch
import librosa
import os
from scipy.signal import butter, filtfilt, medfilt
from scipy.ndimage import uniform_filter1d

from src.analysis.feature_extractor import FeatureExtractor
from src.analysis.key_analyzer import KeyAnalyzer
from src.utils.config import SAMPLE_RATE

from src.ai.mel_encoder import MelEncoder, MelTransitionVAE, ENCODER_DIM
from src.ai.audio_compatibility import AudioCompatibilityScorer


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

        return 1.0 - min(np.std(rms) / (np.mean(rms) + 1e-8), 1.0)


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



class LoopFinder:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.vocal_detector = VocalDetector(sample_rate)
        self.structure_analyzer = StructureAnalyzer(sample_rate)
        self.phrase_detector = PhraseDetector(sample_rate)

    def score_loop(self, audio, start_time, end_time, beat_info, position='any',
                   vocal_scores=None, perc_curve=None):
        score = 0
        reasons = []
        hop = 512

        if vocal_scores is not None:
            sf = max(0, int(start_time * self.sample_rate / hop))
            ef = min(len(vocal_scores), int(end_time * self.sample_rate / hop))
            has_vocals = bool(np.mean(vocal_scores[sf:ef] > 0.45) > 0.40) if ef > sf else False
        else:
            has_vocals = self.vocal_detector.has_vocals_in_range(audio, start_time, end_time, threshold=0.45)

        if has_vocals:
            score -= 70
            reasons.append("VOCALS_DETECTED")
        else:
            score += 40
            reasons.append("NO_VOCALS")

        stability = self.structure_analyzer.get_section_stability(audio, start_time, end_time)
        score += stability * 25
        reasons.append(f"STABILITY_{stability:.0%}")

        if perc_curve is not None:
            sf = max(0, int(start_time * self.sample_rate / hop))
            ef = min(len(perc_curve), int(end_time * self.sample_rate / hop))
            if ef > sf:
                seg = audio[int(start_time * self.sample_rate):int(end_time * self.sample_rate)]
                seg_rms = np.sqrt(np.mean(seg ** 2)) if len(seg) > 0 else 1e-8
                perc_ratio = np.mean(perc_curve[sf:ef]) / (seg_rms + 1e-8)
            else:
                perc_ratio = 0.35
        else:
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            segment = audio[start_sample:end_sample]
            perc_ratio = 0.35
            if len(segment) > self.sample_rate:
                try:
                    _, percussive = librosa.effects.hpss(segment)
                    perc_ratio = np.sqrt(np.mean(percussive ** 2)) / (np.sqrt(np.mean(segment ** 2)) + 1e-8)
                except:
                    pass

        if perc_ratio < 0.30:
            score += 22
            reasons.append("LOW_PERCUSSION")
        elif perc_ratio < 0.45:
            score += 12
            reasons.append("MED_PERCUSSION")
        
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

    def find_best_loop(self, audio, target_time, n_bars=8, search_range=30.0, position='any',
                       beat_info=None, sections=None, instrumental=None):
        duration = len(audio) / self.sample_rate

        if beat_info is None:
            beat_info = self.phrase_detector.detect_beats_and_phrases(audio)
        bar_dur = beat_info['bar_duration']
        loop_dur = bar_dur * n_bars
        bars = beat_info['bars']

        if len(bars) == 0:
            return target_time, target_time + loop_dur, True, []

        if sections is None:
            sections = self.structure_analyzer.detect_sections(audio)
        loopable_sections = [s for s in sections if s.get('loopable', False)]

        if instrumental is None:
            instrumental = self.vocal_detector.get_instrumental_segments(audio, min_duration=loop_dur * 0.8)

        vocal_scores = self.vocal_detector.detect_vocals_curve(audio)
        try:
            _, _percussive = librosa.effects.hpss(audio)
            perc_curve = librosa.feature.rms(y=_percussive, hop_length=512)[0]
        except Exception:
            perc_curve = None
        
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
            
            score, has_vocals, reasons = self.score_loop(
                audio, bar_time, loop_end, beat_info, position,
                vocal_scores=vocal_scores, perc_curve=perc_curve
            )
            
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

    def find_outro_loop(self, audio, n_bars=8, beat_info=None, sections=None, instrumental=None):
        duration = len(audio) / self.sample_rate
        target = duration * 0.75
        return self.find_best_loop(audio, target, n_bars, search_range=duration * 0.22, position='outro',
                                   beat_info=beat_info, sections=sections, instrumental=instrumental)

    def find_intro_loop(self, audio, n_bars=8, beat_info=None, sections=None, instrumental=None):
        duration = len(audio) / self.sample_rate
        target = min(18.0, duration * 0.15)
        return self.find_best_loop(audio, target, n_bars, search_range=22.0, position='intro',
                                   beat_info=beat_info, sections=sections, instrumental=instrumental)

    def find_breakdown_loop(self, audio, n_bars=8, beat_info=None, sections=None, instrumental=None):
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
            search_range = max(best_bd['duration'], 30.0)
            return self.find_best_loop(audio, target, n_bars, search_range=search_range, position='any',
                                       beat_info=beat_info, sections=sections, instrumental=instrumental)

        return self.find_best_loop(audio, duration * 0.5, n_bars, position='any',
                                   beat_info=beat_info, sections=sections, instrumental=instrumental)


class SmartTransitionGenerator:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.feature_extractor = FeatureExtractor(sample_rate)
        self.key_analyzer = KeyAnalyzer(sample_rate)
        self.vocal_detector = VocalDetector(sample_rate)
        self.structure_analyzer = StructureAnalyzer(sample_rate)
        self.phrase_detector = PhraseDetector(sample_rate)
        self.harmonic_mixer = HarmonicMixer(sample_rate)
        self.loop_finder = LoopFinder(sample_rate)
        self.circle_of_fifths = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']
        self.mel_encoder = None
        self.mel_vae = None
        self.compat_scorer = AudioCompatibilityScorer(sample_rate)
        self._load_mel_models()

    def _load_mel_models(self):
        enc_path = "models/mel_encoder.pth"
        vae_path = "models/mel_vae.pth"
        if os.path.exists(enc_path):
            try:
                ckpt = torch.load(enc_path, map_location=self.device)
                self.mel_encoder = MelEncoder(
                    n_mels=ckpt.get('n_mels', 64),
                    embedding_dim=ckpt.get('embedding_dim', ENCODER_DIM),
                ).to(self.device)
                self.mel_encoder.load_state_dict(ckpt['model_state'])
                self.mel_encoder.eval()
                print("  [OK] MelEncoder loaded")
            except Exception as e:
                print(f"  [!] MelEncoder error: {e}")
        if os.path.exists(vae_path) and self.mel_encoder is not None:
            try:
                ckpt = torch.load(vae_path, map_location=self.device)
                self.mel_vae = MelTransitionVAE(
                    embedding_dim=ckpt.get('embedding_dim', ENCODER_DIM * 2),
                    latent_dim=ckpt.get('latent_dim', 64),
                    output_dim=ckpt.get('output_dim', 24),
                ).to(self.device)
                self.mel_vae.load_state_dict(ckpt['model_state'])
                self.mel_vae.eval()
                print("  [OK] MelTransitionVAE loaded")
            except Exception as e:
                print(f"  [!] MelTransitionVAE error: {e}")

    def _get_ai_params(self, audio1, audio2):
        emb1 = self.mel_encoder.embed_audio(audio1, self.sample_rate, self.device)
        emb2 = self.mel_encoder.embed_audio(audio2, self.sample_rate, self.device)
        inp = torch.FloatTensor(np.concatenate([emb1, emb2])).unsqueeze(0).to(self.device)
        params_tensor = self.mel_vae.predict(inp)
        return self.mel_vae.get_params_dict(params_tensor)

    def analyze_track(self, audio, name="Track"):
        print(f"\n  === ANALYSE: {name} ===")
        
        key_info = self.key_analyzer.detect_key(audio)
        beat_info = self.phrase_detector.detect_beats_and_phrases(audio)
        camelot = self.harmonic_mixer.get_camelot(key_info['key'], key_info['mode'])
        
        sections = self.structure_analyzer.detect_sections(audio)
        vocal_segments = self.vocal_detector.get_vocal_segments(audio, threshold=0.45)
        instrumental_segments = self.vocal_detector.get_instrumental_segments(audio, threshold=0.45)
        
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

    def generate_transition(self, audio1, audio2):
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

        print(f"\n  === TEMPO SYNC ===")
        print(f"  Track A: {tempo1:.1f} BPM")
        print(f"  Track B: {tempo2:.1f} BPM")
        print(f"  Mix tempo: {avg_tempo:.1f} BPM")
        
        n_bars = 8 if p['transition_beats'] >= 32 else 4
        
        print(f"\n  === LOOP DETECTION ===")
        print(f"  Target: {n_bars} bars")
        
        print(f"\n  Track A - Finding outro loop...")
        loop1_start, loop1_end, has_v1, reasons1 = self.loop_finder.find_outro_loop(
            audio1, n_bars=n_bars,
            beat_info=analysis1['beat_info'], sections=analysis1['sections'],
            instrumental=analysis1['instrumental_segments']
        )
        print(f"    Position: {loop1_start:.1f}s - {loop1_end:.1f}s ({loop1_end - loop1_start:.1f}s)")
        print(f"    Vocals: {'YES' if has_v1 else 'NO'}")
        print(f"    Criteria: {', '.join(reasons1[:4])}")

        if has_v1:
            print(f"    -> Searching alternative breakdown...")
            loop1_start, loop1_end, has_v1, reasons1 = self.loop_finder.find_breakdown_loop(
                audio1, n_bars=n_bars,
                beat_info=analysis1['beat_info'], sections=analysis1['sections'],
                instrumental=analysis1['instrumental_segments']
            )
            print(f"    New: {loop1_start:.1f}s - {loop1_end:.1f}s | Vocals: {'YES' if has_v1 else 'NO'}")

        print(f"\n  Track B - Finding intro loop...")
        loop2_start, loop2_end, has_v2, reasons2 = self.loop_finder.find_intro_loop(
            audio2, n_bars=n_bars,
            beat_info=analysis2['beat_info'], sections=analysis2['sections'],
            instrumental=analysis2['instrumental_segments']
        )
        print(f"    Position: {loop2_start:.1f}s - {loop2_end:.1f}s ({loop2_end - loop2_start:.1f}s)")
        print(f"    Vocals: {'YES' if has_v2 else 'NO'}")
        print(f"    Criteria: {', '.join(reasons2[:4])}")

        if has_v2:
            print(f"    -> Searching alternative breakdown...")
            loop2_start, loop2_end, has_v2, reasons2 = self.loop_finder.find_breakdown_loop(
                audio2, n_bars=n_bars,
                beat_info=analysis2['beat_info'], sections=analysis2['sections'],
                instrumental=analysis2['instrumental_segments']
            )
            print(f"    New: {loop2_start:.1f}s - {loop2_end:.1f}s | Vocals: {'YES' if has_v2 else 'NO'}")
        
        trans_beats = p['transition_beats']
        trans_dur = (trans_beats / avg_tempo) * 60
        trans_dur = max(12, min(trans_dur, 48))
        
        print(f"\n  === TRANSITION BUILD ===")
        print(f"  Duration: {trans_dur:.1f}s ({trans_beats:.0f} beats)")
        
        n_samples = int(trans_dur * self.sample_rate)

        cf_type = int(round(p.get('crossfade_type', 1)))
        bass_swap_beat = p.get('bass_swap_beat', 0.5)
        print(f"\n  Building AI-driven crossfade...")
        print(f"  Curve type: {cf_type} ({['linear','cosine²','power'][cf_type if cf_type in (0,1,2) else 1]}) | Bass swap: {bass_swap_beat:.0%}")

        seg1_start = int(loop1_start * self.sample_rate)
        seg2_start = int(loop2_start * self.sample_rate)

        seg1 = audio1[seg1_start:seg1_start + n_samples]
        seg2 = audio2[seg2_start:seg2_start + n_samples]

        if len(seg1) < n_samples:
            seg1 = np.pad(seg1, (0, n_samples - len(seg1)))
        if len(seg2) < n_samples:
            seg2 = np.pad(seg2, (0, n_samples - len(seg2)))

        t = np.linspace(0, 1, n_samples)

        if cf_type == 0:
            curve_out = 1 - t
            curve_in = t
        elif cf_type == 1:
            curve_out = np.cos(t * np.pi / 2) ** 2
            curve_in = np.sin(t * np.pi / 2) ** 2
        else:
            curve_out = (1 - t) ** 1.5
            curve_in = t ** 1.5

        try:
            nyq = self.sample_rate / 2
            b_lo, a_lo = butter(4, 200 / nyq, btype='low')
            b_hi, a_hi = butter(4, 200 / nyq, btype='high')

            bass1 = filtfilt(b_lo, a_lo, seg1)
            body1 = filtfilt(b_hi, a_hi, seg1)
            bass2 = filtfilt(b_lo, a_lo, seg2)
            body2 = filtfilt(b_hi, a_hi, seg2)

            bar_samples = int((4 * 60 / avg_tempo) * self.sample_rate)
            swap_idx = int(n_samples * bass_swap_beat)
            swap_idx = (swap_idx // bar_samples) * bar_samples
            swap_idx = max(bar_samples, min(swap_idx, n_samples - bar_samples))

            low_mix = np.zeros(n_samples)
            low_mix[:swap_idx] = bass1[:swap_idx]
            swap_len = min(bar_samples, n_samples - swap_idx)
            if swap_len > 0:
                sw_t = np.linspace(0, 1, swap_len)
                sw_out = np.cos(sw_t * np.pi / 2)
                sw_in = np.sin(sw_t * np.pi / 2)
                low_mix[swap_idx:swap_idx + swap_len] = (
                    bass1[swap_idx:swap_idx + swap_len] * sw_out +
                    bass2[swap_idx:swap_idx + swap_len] * sw_in
                )
            if swap_idx + swap_len < n_samples:
                low_mix[swap_idx + swap_len:] = bass2[swap_idx + swap_len:]

            body_mix = body1 * curve_out + body2 * curve_in
            transition = low_mix + body_mix
            print(f"  Bass swap at: {swap_idx / self.sample_rate:.1f}s into transition")
        except Exception as e:
            print(f"  [!] Bass swap failed ({e}), using full crossfade")
            transition = seg1 * curve_out + seg2 * curve_in

        loop2_end = loop2_start + trans_dur

        peak = np.max(np.abs(transition))
        if peak > 0:
            transition = transition * (0.95 / peak)

        audio_scores = None
        if self.compat_scorer is not None:
            try:
                audio_scores = self.compat_scorer.score(seg1, seg2)
                print(f"\n  === AUDIO COMPATIBILITY (mel analysis) ===")
                print(f"  Total score:        {audio_scores['total_score']:.2%}")
                print(f"  Energy continuity:  {audio_scores['energy_continuity']:.2%}")
                print(f"  Spectral match:     {audio_scores['spectral_match']:.2%}")
                print(f"  Harmonic coherence: {audio_scores['harmonic_coherence']:.2%}")
            except Exception as e:
                print(f"  [!] Compatibility scoring error: {e}")

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
            'params': p,
            'audio_scores': audio_scores,
            'mel_model_used': self.mel_vae is not None,
        }

