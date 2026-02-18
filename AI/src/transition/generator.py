import numpy as np
from src.audio.loader import AudioLoader
from src.audio.exporter import AudioExporter
from src.analysis.feature_extractor import FeatureExtractor
from src.analysis.beat_detector import BeatDetector
from src.analysis.key_analyzer import KeyAnalyzer
from src.ai.smart_transition import SmartTransitionGenerator
from src.utils.config import SAMPLE_RATE


class TransitionGenerator:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.loader = AudioLoader(sample_rate)
        self.exporter = AudioExporter(sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate)
        self.beat_detector = BeatDetector(sample_rate)
        self.key_analyzer = KeyAnalyzer(sample_rate)
        self.ai_generator = SmartTransitionGenerator(sample_rate)

    def generate_transition(self, track1_path, track2_path, transition_duration=20.0,
                           output_path=None, style='auto', overlap_duration=3.5):
        print("\n" + "=" * 60)
        print("  DJ TRANSITION GENERATOR")
        print("=" * 60)
        
        print("\n[1/5] Loading tracks...")
        audio1, _ = self.loader.load(track1_path)
        audio2, _ = self.loader.load(track2_path)
        
        print(f"  Track A: {len(audio1)/self.sample_rate:.1f}s")
        print(f"  Track B: {len(audio2)/self.sample_rate:.1f}s")
        
        print("\n[2/5] AI Analysis & Transition Generation...")
        transition_audio, cut_info = self.ai_generator.generate_transition(
            audio1, audio2,
            duration=transition_duration,
            style=style
        )
        
        loop1_info = cut_info['loop1']
        loop2_info = cut_info['loop2']
        actual_duration = cut_info['transition_duration']
        
        print("\n[3/5] Assembling final mix...")
        
        overlap_samples = int(overlap_duration * self.sample_rate)
        cut_sample1 = int(loop1_info['start'] * self.sample_rate)
        start_sample2 = int(loop2_info['end'] * self.sample_rate)
        
        part1_end = max(0, cut_sample1 - overlap_samples)
        part1 = audio1[:part1_end]
        
        overlap_in_len = min(overlap_samples, cut_sample1, len(transition_audio))
        if overlap_in_len > 0:
            a1_overlap = audio1[part1_end:part1_end + overlap_in_len]
            t_overlap = transition_audio[:overlap_in_len]
            
            min_len = min(len(a1_overlap), len(t_overlap))
            if min_len > 0:
                t = np.linspace(0, 1, min_len)
                part2 = a1_overlap[:min_len] * np.cos(t * np.pi / 2) + t_overlap[:min_len] * np.sin(t * np.pi / 2)
            else:
                part2 = np.array([])
        else:
            part2 = np.array([])
        
        core_start = overlap_in_len
        core_end = len(transition_audio) - overlap_samples
        if core_end > core_start:
            part3 = transition_audio[core_start:core_end]
        else:
            part3 = transition_audio[core_start:]
        
        overlap_out_len = min(overlap_samples, len(transition_audio), len(audio2) - start_sample2)
        
        if overlap_out_len > 0:
            t_out = transition_audio[-overlap_out_len:]
            a2_overlap = audio2[start_sample2:start_sample2 + overlap_out_len]
            
            min_len = min(len(t_out), len(a2_overlap))
            if min_len > 0:
                t = np.linspace(0, 1, min_len)
                part4 = t_out[:min_len] * np.cos(t * np.pi / 2) + a2_overlap[:min_len] * np.sin(t * np.pi / 2)
            else:
                part4 = np.array([])
        else:
            part4 = np.array([])
        
        track2_continue = start_sample2 + overlap_out_len
        part5 = audio2[track2_continue:] if track2_continue < len(audio2) else np.array([])
        
        parts = [p for p in [part1, part2, part3, part4, part5] if len(p) > 0]
        full_mix = np.concatenate(parts) if parts else np.array([])
        
        print("\n[4/5] Normalizing...")
        if len(full_mix) > 0:
            peak = np.max(np.abs(full_mix))
            if peak > 0:
                full_mix = full_mix * (0.95 / peak)
        
        duration_total = len(full_mix) / self.sample_rate
        
        print(f"\n  Assembly breakdown:")
        print(f"    Part 1 (Track A clean): {len(part1)/self.sample_rate:.1f}s")
        print(f"    Part 2 (Fade in):       {len(part2)/self.sample_rate:.1f}s")
        print(f"    Part 3 (Transition):    {len(part3)/self.sample_rate:.1f}s")
        print(f"    Part 4 (Fade out):      {len(part4)/self.sample_rate:.1f}s")
        print(f"    Part 5 (Track B clean): {len(part5)/self.sample_rate:.1f}s")
        print(f"\n  Total duration: {duration_total:.1f}s")
        
        if output_path:
            print(f"\n[5/5] Exporting...")
            self.exporter.export_wav(full_mix, output_path)
            print(f"  Saved: {output_path}")
        
        print("\n" + "=" * 60)
        print("  TRANSITION COMPLETE")
        print("=" * 60)
        
        return {
            'audio': full_mix,
            'duration': duration_total,
            'transition_duration': actual_duration,
            'cut_info': cut_info,
            'harmony': cut_info['analysis']['harmony']
        }
