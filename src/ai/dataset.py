"""Module de création et gestion du dataset pour l'entraînement."""

import os
import numpy as np
import librosa
from pathlib import Path
from typing import List, Tuple
import pickle
import random

from src.utils.config import SAMPLE_RATE, HOP_LENGTH, N_FFT


class TransitionDataset:
    """
    Crée un dataset de transitions pour l'entraînement du modèle.
    
    Le dataset contient des triplets :
    - Spectrogramme de fin de morceau (input 1)
    - Spectrogramme de début de morceau suivant (input 2)
    - Spectrogramme de la zone de transition idéale (target)
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, 
                 n_mels: int = 128,
                 segment_duration: float = 5.0):
        """
        Args:
            sample_rate: Fréquence d'échantillonnage
            n_mels: Nombre de bandes mel
            segment_duration: Durée des segments en secondes
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = N_FFT
        self.hop_length = HOP_LENGTH
        self.segment_duration = segment_duration
        self.segment_samples = int(segment_duration * sample_rate)
        
        # Données du dataset
        self.data = []
    
    def audio_to_melspec(self, audio: np.ndarray) -> np.ndarray:
        """Convertit audio en spectrogramme mel normalisé."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convertir en dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normaliser entre 0 et 1
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        return mel_spec_norm
    
    def melspec_to_audio(self, mel_spec_norm: np.ndarray, 
                         ref_db_min: float = -80.0, 
                         ref_db_max: float = 0.0) -> np.ndarray:
        """Convertit spectrogramme mel en audio."""
        # Dénormaliser
        mel_spec_db = mel_spec_norm * (ref_db_max - ref_db_min) + ref_db_min
        
        # Convertir de dB vers puissance
        mel_spec = librosa.db_to_power(mel_spec_db)
        
        # Inverser le mel spectrogram
        audio = librosa.feature.inverse.mel_to_audio(
            mel_spec,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return audio
    
    def create_training_sample(self, audio1: np.ndarray, audio2: np.ndarray) -> dict:
        """
        Crée un échantillon d'entraînement à partir de deux morceaux.
        
        La "transition idéale" est simulée en faisant un crossfade
        de haute qualité entre les deux morceaux.
        
        Args:
            audio1: Premier morceau
            audio2: Deuxième morceau
            
        Returns:
            dict: Échantillon avec input1, input2, target
        """
        # Extraire la fin du morceau 1
        if len(audio1) > self.segment_samples:
            end1 = audio1[-self.segment_samples:]
        else:
            end1 = np.pad(audio1, (self.segment_samples - len(audio1), 0))
        
        # Extraire le début du morceau 2
        if len(audio2) > self.segment_samples:
            start2 = audio2[:self.segment_samples]
        else:
            start2 = np.pad(audio2, (0, self.segment_samples - len(audio2)))
        
        # Créer la transition "idéale" (crossfade equal power)
        t = np.linspace(0, 1, self.segment_samples)
        fade_out = np.cos(t * np.pi / 2)
        fade_in = np.sin(t * np.pi / 2)
        
        transition = end1 * fade_out + start2 * fade_in
        
        # Convertir en spectrogrammes
        spec1 = self.audio_to_melspec(end1)
        spec2 = self.audio_to_melspec(start2)
        spec_transition = self.audio_to_melspec(transition)
        
        # S'assurer que tous les spectrogrammes ont la même taille
        min_frames = min(spec1.shape[1], spec2.shape[1], spec_transition.shape[1])
        
        spec1 = spec1[:, :min_frames]
        spec2 = spec2[:, :min_frames]
        spec_transition = spec_transition[:, :min_frames]
        
        return {
            'input1': spec1,      # Fin morceau 1
            'input2': spec2,      # Début morceau 2
            'target': spec_transition  # Transition idéale
        }
    
    def build_from_folder(self, music_folder: str, max_samples: int = 100):
        """
        Construit le dataset à partir d'un dossier de musiques.
        
        Args:
            music_folder: Chemin vers le dossier contenant les musiques
            max_samples: Nombre maximum d'échantillons à créer
        """
        print(f"📂 Scan du dossier: {music_folder}")
        
        # Trouver tous les fichiers audio
        audio_files = []
        for ext in ['*.mp3', '*.wav', '*.flac', '*.ogg']:
            audio_files.extend(Path(music_folder).glob(ext))
        
        audio_files = list(audio_files)
        print(f"  Fichiers trouvés: {len(audio_files)}")
        
        if len(audio_files) < 2:
            print("  ⚠️ Il faut au moins 2 fichiers audio !")
            return
        
        # Charger tous les audios
        print("🎵 Chargement des fichiers audio...")
        audios = []
        for f in audio_files:
            try:
                audio, _ = librosa.load(str(f), sr=self.sample_rate, mono=True)
                if len(audio) > self.segment_samples * 2:
                    audios.append(audio)
                    print(f"  ✓ {f.name}")
            except Exception as e:
                print(f"  ✗ {f.name}: {e}")
        
        print(f"\n📊 Fichiers chargés: {len(audios)}")
        
        # Créer les échantillons
        print(f"\n🔨 Création des échantillons d'entraînement...")
        
        samples_created = 0
        
        for i in range(min(max_samples, len(audios) * (len(audios) - 1))):
            # Choisir deux morceaux aléatoires différents
            idx1, idx2 = random.sample(range(len(audios)), 2)
            
            # Créer l'échantillon
            sample = self.create_training_sample(audios[idx1], audios[idx2])
            self.data.append(sample)
            
            samples_created += 1
            
            if samples_created % 10 == 0:
                print(f"  Échantillons créés: {samples_created}/{max_samples}")
        
        print(f"\n✅ Dataset créé: {len(self.data)} échantillons")
    
    def augment_data(self):
        """Augmente le dataset avec des variations."""
        print("🔄 Augmentation du dataset...")
        
        original_size = len(self.data)
        augmented = []
        
        for sample in self.data:
            # 1. Inverser les entrées (créer transition dans l'autre sens)
            augmented.append({
                'input1': sample['input2'],
                'input2': sample['input1'],
                'target': sample['target'][:, ::-1]  # Inverser temporellement
            })
            
            # 2. Ajouter du bruit léger
            noise_level = 0.02
            augmented.append({
                'input1': sample['input1'] + np.random.randn(*sample['input1'].shape) * noise_level,
                'input2': sample['input2'] + np.random.randn(*sample['input2'].shape) * noise_level,
                'target': sample['target']
            })
        
        self.data.extend(augmented)
        print(f"  Dataset augmenté: {original_size} → {len(self.data)} échantillons")
    
    def save(self, path: str):
        """Sauvegarde le dataset."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'data': self.data,
                'config': {
                    'sample_rate': self.sample_rate,
                    'n_mels': self.n_mels,
                    'segment_duration': self.segment_duration
                }
            }, f)
        
        print(f"💾 Dataset sauvegardé: {path}")
    
    def load(self, path: str):
        """Charge le dataset."""
        with open(path, 'rb') as f:
            saved = pickle.load(f)
        
        self.data = saved['data']
        config = saved['config']
        self.sample_rate = config['sample_rate']
        self.n_mels = config['n_mels']
        self.segment_duration = config['segment_duration']
        
        print(f"📂 Dataset chargé: {len(self.data)} échantillons")
    
    def get_batch(self, batch_size: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retourne un batch pour l'entraînement.
        
        Returns:
            Tuple: (input1_batch, input2_batch, target_batch)
        """
        indices = random.sample(range(len(self.data)), min(batch_size, len(self.data)))
        
        input1_batch = np.array([self.data[i]['input1'] for i in indices])
        input2_batch = np.array([self.data[i]['input2'] for i in indices])
        target_batch = np.array([self.data[i]['target'] for i in indices])
        
        return input1_batch, input2_batch, target_batch
    
    def __len__(self):
        return len(self.data)