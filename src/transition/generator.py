"""Module générateur de transitions - orchestre le processus complet."""

import numpy as np
from src.audio.loader import AudioLoader
from src.audio.exporter import AudioExporter
from src.analysis.feature_extractor import FeatureExtractor
from src.analysis.beat_detector import BeatDetector
from src.analysis.key_analyzer import KeyAnalyzer
from src.ai.smart_transition import SmartTransitionGenerator
from src.utils.config import SAMPLE_RATE


class TransitionGenerator:
    """Classe principale pour générer des transitions entre morceaux."""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        
        self.loader = AudioLoader(sample_rate)
        self.exporter = AudioExporter(sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate)
        self.beat_detector = BeatDetector(sample_rate)
        self.key_analyzer = KeyAnalyzer(sample_rate)
        
        # Générateur IA intelligent
        self.ai_generator = SmartTransitionGenerator(sample_rate)
    
    def analyze_track(self, audio: np.ndarray) -> dict:
        """Analyse complète d'un morceau."""
        features = self.feature_extractor.extract_all(audio)
        key_info = self.key_analyzer.detect_key(audio)
        outro_point = self.beat_detector.get_best_outro_point(audio)
        intro_point = self.beat_detector.get_best_intro_point(audio)
        
        return {
            **features,
            'key': key_info['key'],
            'mode': key_info['mode'],
            'key_confidence': key_info['confidence'],
            'best_outro': outro_point,
            'best_intro': intro_point
        }
    
    def _create_crossfade_curve(self, length: int, curve_type: str = 'equal_power') -> np.ndarray:
        """Crée une courbe de crossfade."""
        t = np.linspace(0, 1, length)
        
        if curve_type == 'equal_power':
            return np.sin(t * np.pi / 2)
        elif curve_type == 's_curve':
            return t * t * (3 - 2 * t)
        else:
            return t
    
    def generate_transition(self, track1_path: str, track2_path: str,
                            transition_duration: float = 15.0,
                            output_path: str = None,
                            style: str = 'smooth',
                            overlap_duration: float = 3.0) -> dict:
        """
        Génère une transition entre deux morceaux.
        
        La transition CHEVAUCHE les morceaux pour un enchaînement naturel :
        - La fin du morceau 1 se fond dans le début de la transition
        - La fin de la transition se fond dans le début du morceau 2
        
        Args:
            track1_path: Chemin du premier morceau
            track2_path: Chemin du deuxième morceau
            transition_duration: Durée de la transition générée (secondes)
            output_path: Chemin de sortie
            style: Style de transition ('smooth', 'drop', 'echo')
            overlap_duration: Durée du chevauchement de chaque côté (secondes)
        """
        print("\n" + "=" * 60)
        print("🎵 GÉNÉRATION DE TRANSITION IA")
        print("=" * 60)
        
        # === Étape 1: Charger les morceaux ===
        print("\n📂 Chargement des morceaux...")
        audio1, _ = self.loader.load(track1_path)
        audio2, _ = self.loader.load(track2_path)
        
        # === Étape 2: Analyser les morceaux ===
        print("\n🔍 Analyse du morceau 1...")
        analysis1 = self.analyze_track(audio1)
        print(f"  - BPM: {analysis1['bpm']}, Key: {analysis1['key']} {analysis1['mode']}")
        print(f"  - Énergie: {analysis1['energy']:.2f}")
        print(f"  - Durée: {analysis1['duration']:.2f}s")
        
        print("\n🔍 Analyse du morceau 2...")
        analysis2 = self.analyze_track(audio2)
        print(f"  - BPM: {analysis2['bpm']}, Key: {analysis2['key']} {analysis2['mode']}")
        print(f"  - Énergie: {analysis2['energy']:.2f}")
        print(f"  - Durée: {analysis2['duration']:.2f}s")
        
        # === Étape 3: Trouver les points de coupe ===
        print("\n✂️ Points de transition optimaux...")
        
        # Point où commence la transition dans le morceau 1
        outro_time = analysis1['best_outro']['time']
        print(f"  - Début transition (morceau 1): {outro_time:.2f}s")
        
        # Point où la transition se termine dans le morceau 2
        intro_time = analysis2['best_intro']['time']
        print(f"  - Fin transition (morceau 2): {intro_time:.2f}s")
        
        # === Étape 4: Générer la transition IA ===
        print(f"\n🤖 Génération IA de la transition ({transition_duration}s, style: {style})...")
        
        # L'IA utilise la fin du morceau 1 et le début du morceau 2
        # pour créer un pont sonore cohérent
        transition_audio = self.ai_generator.generate_transition(
            audio1,  # Morceau 1 complet (l'IA prend la fin)
            audio2,  # Morceau 2 complet (l'IA prend le début)
            duration=transition_duration,
            style=style
        )
        
        # === Étape 5: Assembler avec CHEVAUCHEMENT ===
        print("\n🎛️ Assemblage avec chevauchement...")
        print(f"  - Durée de chevauchement: {overlap_duration}s de chaque côté")
        
        overlap_samples = int(overlap_duration * self.sample_rate)
        outro_idx = int(outro_time * self.sample_rate)
        intro_idx = int(intro_time * self.sample_rate)
        
        # ===== PARTIE 1: Morceau 1 jusqu'au début du chevauchement =====
        # On garde le morceau 1 jusqu'à (outro_time - overlap_duration)
        cut_point_1 = max(0, outro_idx - overlap_samples)
        part1_clean = audio1[:cut_point_1]
        
        # ===== PARTIE 2: Zone de chevauchement ENTRÉE (Morceau 1 + Transition) =====
        # Morceau 1 : de cut_point_1 à outro_idx (fade out)
        overlap_in_track1 = audio1[cut_point_1:outro_idx]
        
        # Transition : le début (fade in)
        overlap_in_transition = transition_audio[:len(overlap_in_track1)]
        
        # S'assurer qu'ils ont la même taille
        min_len_in = min(len(overlap_in_track1), len(overlap_in_transition))
        if min_len_in > 0:
            overlap_in_track1 = overlap_in_track1[:min_len_in]
            overlap_in_transition = overlap_in_transition[:min_len_in]
            
            # Créer les courbes de crossfade
            fade_out_curve = 1 - self._create_crossfade_curve(min_len_in, 'equal_power')
            fade_in_curve = self._create_crossfade_curve(min_len_in, 'equal_power')
            
            # Mixer
            part2_overlap_in = overlap_in_track1 * fade_out_curve + overlap_in_transition * fade_in_curve
        else:
            part2_overlap_in = np.array([])
        
        # ===== PARTIE 3: Cœur de la transition (sans chevauchement) =====
        transition_start = len(overlap_in_transition) if len(overlap_in_transition) > 0 else 0
        transition_end = len(transition_audio) - overlap_samples
        
        if transition_end > transition_start:
            part3_transition_core = transition_audio[transition_start:transition_end]
        else:
            part3_transition_core = np.array([])
        
        # ===== PARTIE 4: Zone de chevauchement SORTIE (Transition + Morceau 2) =====
        # Transition : la fin (fade out)
        overlap_out_transition = transition_audio[-overlap_samples:] if len(transition_audio) >= overlap_samples else transition_audio
        
        # Morceau 2 : de intro_idx à (intro_idx + overlap_duration) (fade in)
        overlap_out_track2 = audio2[intro_idx:intro_idx + len(overlap_out_transition)]
        
        # S'assurer qu'ils ont la même taille
        min_len_out = min(len(overlap_out_transition), len(overlap_out_track2))
        if min_len_out > 0:
            overlap_out_transition = overlap_out_transition[:min_len_out]
            overlap_out_track2 = overlap_out_track2[:min_len_out]
            
            # Créer les courbes de crossfade
            fade_out_curve = 1 - self._create_crossfade_curve(min_len_out, 'equal_power')
            fade_in_curve = self._create_crossfade_curve(min_len_out, 'equal_power')
            
            # Mixer
            part4_overlap_out = overlap_out_transition * fade_out_curve + overlap_out_track2 * fade_in_curve
        else:
            part4_overlap_out = np.array([])
        
        # ===== PARTIE 5: Morceau 2 après le chevauchement =====
        cut_point_2 = intro_idx + min_len_out if min_len_out > 0 else intro_idx
        part5_clean = audio2[cut_point_2:]
        
        # ===== ASSEMBLER TOUTES LES PARTIES =====
        parts = []
        
        if len(part1_clean) > 0:
            parts.append(part1_clean)
            print(f"  ✓ Morceau 1 (clean): {len(part1_clean) / self.sample_rate:.2f}s")
        
        if len(part2_overlap_in) > 0:
            parts.append(part2_overlap_in)
            print(f"  ✓ Chevauchement entrée: {len(part2_overlap_in) / self.sample_rate:.2f}s")
        
        if len(part3_transition_core) > 0:
            parts.append(part3_transition_core)
            print(f"  ✓ Transition (cœur): {len(part3_transition_core) / self.sample_rate:.2f}s")
        
        if len(part4_overlap_out) > 0:
            parts.append(part4_overlap_out)
            print(f"  ✓ Chevauchement sortie: {len(part4_overlap_out) / self.sample_rate:.2f}s")
        
        if len(part5_clean) > 0:
            parts.append(part5_clean)
            print(f"  ✓ Morceau 2 (clean): {len(part5_clean) / self.sample_rate:.2f}s")
        
        # Concaténer
        full_mix = np.concatenate(parts)
        
        duration_total = len(full_mix) / self.sample_rate
        print(f"\n  📊 Durée totale: {duration_total:.2f}s")
        
        # === Étape 6: Exporter ===
        if output_path:
            self.exporter.export_wav(full_mix, output_path)
        
        print("\n" + "=" * 60)
        print("✅ TRANSITION GÉNÉRÉE AVEC SUCCÈS !")
        print("=" * 60)
        
        return {
            'audio': full_mix,
            'transition_audio': transition_audio,
            'duration': duration_total,
            'transition_duration': len(transition_audio) / self.sample_rate,
            'analysis_track1': analysis1,
            'analysis_track2': analysis2,
            'outro_time': outro_time,
            'intro_time': intro_time,
            'style': style,
            'overlap_duration': overlap_duration
        }