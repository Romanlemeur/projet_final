"""Générateur de transitions - utilise les points de coupe de l'IA."""

import numpy as np
from src.audio.loader import AudioLoader
from src.audio.exporter import AudioExporter
from src.analysis.feature_extractor import FeatureExtractor
from src.analysis.beat_detector import BeatDetector
from src.analysis.key_analyzer import KeyAnalyzer
from src.ai.smart_transition import SmartTransitionGenerator
from src.utils.config import SAMPLE_RATE


class TransitionGenerator:
    """Génère des transitions avec points de coupe décidés par l'IA."""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        
        self.loader = AudioLoader(sample_rate)
        self.exporter = AudioExporter(sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate)
        self.beat_detector = BeatDetector(sample_rate)
        self.key_analyzer = KeyAnalyzer(sample_rate)
        
        self.ai_generator = SmartTransitionGenerator(sample_rate)
    
    def generate_transition(self, track1_path: str, track2_path: str,
                            transition_duration: float = 20.0,
                            output_path: str = None,
                            style: str = 'auto',
                            overlap_duration: float = 5.0) -> dict:
        """
        Génère une transition avec tous les paramètres décidés par l'IA.
        """
        print("\n" + "=" * 60)
        print("🎵 GÉNÉRATION DE TRANSITION IA COMPLÈTE")
        print("=" * 60)
        
        # Charger
        print("Chargement...")
        audio1, _ = self.loader.load(track1_path)
        audio2, _ = self.loader.load(track2_path)
        
        print(f"  Track 1: {len(audio1)/self.sample_rate:.2f}s")
        print(f"  Track 2: {len(audio2)/self.sample_rate:.2f}s")
        
        # L'IA génère la transition ET décide des points de coupe
        transition_audio, cut_info = self.ai_generator.generate_transition(
            audio1, audio2,
            duration=transition_duration,
            style=style
        )
        
        # Récupérer les points de coupe décidés par l'IA
        cut_time1 = cut_info['track1_cut_time']
        start_time2 = cut_info['track2_start_time']
        actual_duration = cut_info['transition_duration']
        
        # === ASSEMBLAGE FINAL ===
        print(f"\n🎛️ Assemblage final...")
        
        overlap_samples = int(overlap_duration * self.sample_rate)
        cut_sample1 = int(cut_time1 * self.sample_rate)
        start_sample2 = int(start_time2 * self.sample_rate)
        
        # Partie 1: Track 1 jusqu'au début du chevauchement
        fade_start1 = max(0, cut_sample1 - overlap_samples)
        part1 = audio1[:fade_start1]
        
        # Partie 2: Chevauchement entrée
        overlap_in_audio1 = audio1[fade_start1:cut_sample1]
        overlap_in_trans = transition_audio[:len(overlap_in_audio1)]
        
        min_len_in = min(len(overlap_in_audio1), len(overlap_in_trans))
        if min_len_in > 0:
            t = np.linspace(0, 1, min_len_in)
            fade_out = np.cos(t * np.pi / 2)
            fade_in = np.sin(t * np.pi / 2)
            part2 = overlap_in_audio1[:min_len_in] * fade_out + overlap_in_trans[:min_len_in] * fade_in
        else:
            part2 = np.array([])
            min_len_in = 0
        
        # Partie 3: Cœur de la transition
        trans_core_start = min_len_in
        trans_core_end = len(transition_audio) - overlap_samples
        
        if trans_core_end > trans_core_start:
            part3 = transition_audio[trans_core_start:trans_core_end]
        else:
            part3 = transition_audio[trans_core_start:]
        
        # Partie 4: Chevauchement sortie
        trans_samples_used = int(actual_duration * self.sample_rate)
        track2_resume = start_sample2 + max(0, trans_samples_used - overlap_samples)
        
        overlap_out_trans = transition_audio[-overlap_samples:] if len(transition_audio) >= overlap_samples else transition_audio
        overlap_out_audio2 = audio2[track2_resume:track2_resume + len(overlap_out_trans)]
        
        min_len_out = min(len(overlap_out_trans), len(overlap_out_audio2))
        if min_len_out > 0:
            t = np.linspace(0, 1, min_len_out)
            fade_out = np.cos(t * np.pi / 2)
            fade_in = np.sin(t * np.pi / 2)
            part4 = overlap_out_trans[:min_len_out] * fade_out + overlap_out_audio2[:min_len_out] * fade_in
        else:
            part4 = np.array([])
            min_len_out = 0
        
        # Partie 5: Track 2 après le chevauchement
        track2_continue = track2_resume + min_len_out
        part5 = audio2[track2_continue:]
        
        # Affichage
        print(f"  ✓ Track 1 (clean): {len(part1)/self.sample_rate:.2f}s")
        print(f"  ✓ Overlap entrée: {len(part2)/self.sample_rate:.2f}s")
        print(f"  ✓ Transition: {len(part3)/self.sample_rate:.2f}s")
        print(f"  ✓ Overlap sortie: {len(part4)/self.sample_rate:.2f}s")
        print(f"  ✓ Track 2 reprend à: {track2_continue/self.sample_rate:.2f}s")
        print(f"  ✓ Track 2 (clean): {len(part5)/self.sample_rate:.2f}s")
        
        # Assembler
        parts = [p for p in [part1, part2, part3, part4, part5] if len(p) > 0]
        full_mix = np.concatenate(parts)
        
        # Normaliser
        peak = np.max(np.abs(full_mix))
        if peak > 0:
            full_mix = full_mix * (0.95 / peak)
        
        duration_total = len(full_mix) / self.sample_rate
        print(f"\n  📊 Durée totale: {duration_total:.2f}s")
        
        # Exporter
        if output_path:
            self.exporter.export_wav(full_mix, output_path)
        
        print("\n" + "=" * 60)
        print("✅ TRANSITION GÉNÉRÉE PAR L'IA !")
        print("=" * 60)
        
        return {
            'audio': full_mix,
            'duration': duration_total,
            'cut_info': cut_info
        }