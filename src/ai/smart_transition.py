"""Générateur de transition intelligent avec modèle VAE."""

import numpy as np
import torch
import librosa
import os

from src.ai.audio_separator import AudioSeparator
from src.ai.effects import AudioEffects
from src.ai.vae_model import TransitionVAE
from src.utils.config import SAMPLE_RATE, HOP_LENGTH, N_FFT


class SmartTransitionGenerator:
    """
    Génère des transitions musicales avec IA (VAE).
    
    Si le modèle VAE est disponible, l'utilise.
    Sinon, utilise la méthode de fallback (séparation + effets).
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, 
                 model_path: str = "models/transition_vae.pth"):
        self.sample_rate = sample_rate
        self.n_mels = 128
        self.hop_length = HOP_LENGTH
        self.n_fft = N_FFT
        
        self.separator = AudioSeparator(sample_rate)
        self.effects = AudioEffects(sample_rate)
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Charger le modèle VAE si disponible
        self.vae_model = None
        self.model_path = model_path
        
        if os.path.exists(model_path):
            self._load_vae_model(model_path)
        else:
            print(f"  ⚠️ Modèle VAE non trouvé: {model_path}")
            print(f"  ↩️ Utilisation du mode fallback (effets)")
    
    def _load_vae_model(self, path: str):
        """Charge le modèle VAE."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.vae_model = TransitionVAE(
                n_mels=checkpoint['n_mels'],
                n_frames=checkpoint['n_frames'],
                latent_dim=checkpoint['latent_dim']
            ).to(self.device)
            
            self.vae_model.load_state_dict(checkpoint['model_state_dict'])
            self.vae_model.eval()
            
            self.n_mels = checkpoint['n_mels']
            self.n_frames = checkpoint['n_frames']
            
            print(f"  ✓ Modèle VAE chargé: {path}")
            print(f"    - Dimensions: {self.n_mels} mels x {self.n_frames} frames")
            
        except Exception as e:
            print(f"  ✗ Erreur chargement VAE: {e}")
            self.vae_model = None
    
    def _audio_to_melspec(self, audio: np.ndarray) -> np.ndarray:
        """Convertit audio en spectrogramme mel normalisé."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normaliser
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        return mel_spec_norm, mel_spec_db.min(), mel_spec_db.max()
    
    def _melspec_to_audio(self, mel_spec_norm: np.ndarray, 
                          db_min: float, db_max: float) -> np.ndarray:
        """Convertit spectrogramme en audio."""
        # Dénormaliser
        mel_spec_db = mel_spec_norm * (db_max - db_min) + db_min
        mel_spec = librosa.db_to_power(mel_spec_db)
        
        # Inverser
        audio = librosa.feature.inverse.mel_to_audio(
            mel_spec,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_iter=32
        )
        
        return audio
    
    def generate_transition(self, audio1: np.ndarray, audio2: np.ndarray,
                            duration: float = 15.0,
                            style: str = 'smooth') -> np.ndarray:
        """
        Génère une transition entre deux morceaux.
        
        Args:
            audio1: Premier morceau (on utilise la fin)
            audio2: Deuxième morceau (on utilise le début)
            duration: Durée de la transition en secondes
            style: Style de transition ('smooth', 'drop', 'echo')
            
        Returns:
            np.ndarray: Audio de la transition
        """
        print(f"  🎛️ Style: {style}")
        
        # Si le modèle VAE est disponible, l'utiliser
        if self.vae_model is not None:
            print(f"  🤖 Génération avec modèle VAE...")
            return self._generate_with_vae(audio1, audio2, duration)
        else:
            print(f"  🔧 Génération avec méthode fallback...")
            return self._generate_fallback(audio1, audio2, duration, style)
    
    def _generate_with_vae(self, audio1: np.ndarray, audio2: np.ndarray,
                           duration: float) -> np.ndarray:
        """Génère une transition avec le modèle VAE."""
        
        # Calculer le nombre d'échantillons pour le segment d'entrée
        segment_samples = int(5.0 * self.sample_rate)  # 5 secondes
        
        # Extraire les segments
        if len(audio1) > segment_samples:
            segment1 = audio1[-segment_samples:]
        else:
            segment1 = np.pad(audio1, (segment_samples - len(audio1), 0))
        
        if len(audio2) > segment_samples:
            segment2 = audio2[:segment_samples]
        else:
            segment2 = np.pad(audio2, (0, segment_samples - len(audio2)))
        
        # Convertir en spectrogrammes
        spec1, db_min1, db_max1 = self._audio_to_melspec(segment1)
        spec2, db_min2, db_max2 = self._audio_to_melspec(segment2)
        
        # Ajuster la taille des spectrogrammes
        target_frames = self.n_frames
        
        spec1 = self._resize_spectrogram(spec1, target_frames)
        spec2 = self._resize_spectrogram(spec2, target_frames)
        
        # Convertir en tenseurs
        spec1_tensor = torch.FloatTensor(spec1).unsqueeze(0).unsqueeze(0).to(self.device)
        spec2_tensor = torch.FloatTensor(spec2).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Générer avec le VAE
        with torch.no_grad():
            transition_spec = self.vae_model.generate(spec1_tensor, spec2_tensor)
        
        # Convertir en numpy
        transition_spec = transition_spec.squeeze().cpu().numpy()
        
        # Convertir en audio
        db_min = (db_min1 + db_min2) / 2
        db_max = (db_max1 + db_max2) / 2
        
        transition_audio = self._melspec_to_audio(transition_spec, db_min, db_max)
        
        # Ajuster la durée
        target_samples = int(duration * self.sample_rate)
        
        if len(transition_audio) < target_samples:
            # Étirer l'audio
            transition_audio = librosa.effects.time_stretch(
                transition_audio, 
                rate=len(transition_audio) / target_samples
            )
        
        transition_audio = transition_audio[:target_samples]
        
        # Appliquer des effets pour améliorer la qualité
        transition_audio = self.effects.apply_fade(transition_audio, 'in', 0.5, 'exponential')
        transition_audio = self.effects.apply_fade(transition_audio, 'out', 0.5, 'exponential')
        
        # Normaliser
        transition_audio = self._normalize(transition_audio)
        
        print(f"  ✓ Transition VAE générée: {len(transition_audio) / self.sample_rate:.2f}s")
        
        return transition_audio
    
    def _generate_fallback(self, audio1: np.ndarray, audio2: np.ndarray,
                           duration: float, style: str) -> np.ndarray:
        """Génère une transition avec la méthode fallback (effets)."""
        
        n_samples = int(duration * self.sample_rate)
        segment_samples = min(n_samples, len(audio1), len(audio2))
        
        segment1 = audio1[-segment_samples:]
        segment2 = audio2[:segment_samples]
        
        # Séparer les composantes
        components1 = self.separator.full_separation(segment1)
        components2 = self.separator.full_separation(segment2)
        
        # Générer selon le style
        if style == 'smooth':
            transition = self._create_smooth_transition(components1, components2, n_samples)
        elif style == 'drop':
            transition = self._create_drop_transition(components1, components2, n_samples)
        elif style == 'echo':
            transition = self._create_echo_transition(components1, components2, n_samples)
        else:
            transition = self._create_smooth_transition(components1, components2, n_samples)
        
        return self._normalize(transition)
    
    def _resize_spectrogram(self, spec: np.ndarray, target_frames: int) -> np.ndarray:
        """Redimensionne un spectrogramme."""
        from scipy.ndimage import zoom
        
        current_frames = spec.shape[1]
        if current_frames == target_frames:
            return spec
        
        zoom_factor = target_frames / current_frames
        return zoom(spec, (1, zoom_factor), order=1)
    
    def _normalize(self, audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """Normalise l'audio."""
        peak = np.max(np.abs(audio))
        if peak > 0:
            return audio * (target_peak / peak)
        return audio
    
    def _create_smooth_transition(self, comp1: dict, comp2: dict, 
                                   n_samples: int) -> np.ndarray:
        """Crée une transition douce (fallback)."""
        
        def resize(arr, target_len):
            if len(arr) >= target_len:
                return arr[:target_len]
            return np.pad(arr, (0, target_len - len(arr)))
        
        perc1 = resize(comp1['percussive'], n_samples)
        perc2 = resize(comp2['percussive'], n_samples)
        harm1 = resize(comp1['harmonic'], n_samples)
        harm2 = resize(comp2['harmonic'], n_samples)
        bass1 = resize(comp1['bass'], n_samples)
        bass2 = resize(comp2['bass'], n_samples)
        
        t = np.linspace(0, 1, n_samples)
        
        drum_out = 1 - (t * t * (3 - 2 * t))
        drum_in = t * t * (3 - 2 * t)
        harm_out = np.cos(t * np.pi / 2)
        harm_in = np.sin(t * np.pi / 2)
        bass_out = 1 - t
        bass_in = t
        
        harm1_filtered = self.effects.apply_filter_sweep(harm1, 8000, 500, 'low')
        harm2_filtered = self.effects.apply_filter_sweep(harm2, 500, 8000, 'low')
        
        drums_mix = perc1 * drum_out + perc2 * drum_in
        harm_mix = harm1_filtered * harm_out + harm2_filtered * harm_in
        bass_mix = bass1 * bass_out + bass2 * bass_in
        
        return drums_mix * 0.35 + harm_mix * 0.45 + bass_mix * 0.20
    
    def _create_drop_transition(self, comp1: dict, comp2: dict, n_samples: int) -> np.ndarray:
        """Crée une transition drop (fallback)."""
        
        def resize(arr, target_len):
            if len(arr) >= target_len:
                return arr[:target_len]
            return np.pad(arr, (0, target_len - len(arr)))
        
        drop_point = int(n_samples * 0.7)
        silence_duration = int(0.1 * self.sample_rate)
        
        buildup = resize(comp1['full'], drop_point)
        buildup = self.effects.apply_filter_sweep(buildup, 8000, 200, 'low')
        buildup = self.effects.apply_fade(buildup, 'out', 0.3, 'exponential')
        
        silence = np.zeros(silence_duration)
        
        drop_length = n_samples - drop_point - silence_duration
        drop = resize(comp2['full'], drop_length)
        drop = self.effects.apply_fade(drop, 'in', 0.1, 'exponential')
        
        transition = np.concatenate([buildup, silence, drop])
        return resize(transition, n_samples)
    
    def _create_echo_transition(self, comp1: dict, comp2: dict, n_samples: int) -> np.ndarray:
        """Crée une transition echo (fallback)."""
        
        def resize(arr, target_len):
            if len(arr) >= target_len:
                return arr[:target_len]
            return np.pad(arr, (0, target_len - len(arr)))
        
        full1 = resize(comp1['full'], n_samples)
        full2 = resize(comp2['full'], n_samples)
        
        t = np.linspace(0, 1, n_samples)
        
        echo1 = self.effects.apply_echo(full1, delay_ms=300, feedback=0.5, mix=0.4)
        echo1 = self.effects.apply_reverb_simple(echo1, decay=0.4, delay_ms=50)
        
        reverb2 = self.effects.apply_reverb_simple(full2, decay=0.3, delay_ms=40)
        
        transition = echo1 * (1 - t) + reverb2 * t
        
        transition = self.effects.apply_fade(transition, 'in', 0.05, 'linear')
        transition = self.effects.apply_fade(transition, 'out', 0.05, 'linear')
        
        return transition


def generate_smart_transition(audio1: np.ndarray, audio2: np.ndarray,
                               duration: float = 15.0,
                               style: str = 'smooth') -> np.ndarray:
    """Fonction utilitaire."""
    generator = SmartTransitionGenerator()
    return generator.generate_transition(audio1, audio2, duration, style)