"""Module d'analyse de la tonalité (key)."""

import numpy as np
import librosa
from src.utils.config import SAMPLE_RATE


class KeyAnalyzer:
    """Classe pour analyser la tonalité d'un morceau."""
    
    # Noms des tonalités
    KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Cercle des quintes pour compatibilité
    CIRCLE_OF_FIFTHS = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        """
        Initialise l'analyseur de tonalité.
        
        Args:
            sample_rate: Fréquence d'échantillonnage (Hz)
        """
        self.sample_rate = sample_rate
    
    def detect_key(self, audio: np.ndarray) -> dict:
        """
        Détecte la tonalité du morceau.
        
        Args:
            audio: Signal audio
            
        Returns:
            dict: Informations sur la tonalité
        """
        # Extraire le chromagramme
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sample_rate)
        
        # Moyenne sur le temps
        chroma_mean = np.mean(chroma, axis=1)
        
        # Profils de référence pour majeur et mineur (Krumhansl-Schmuckler)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 
                                   2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 
                                   2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Normaliser
        major_profile = major_profile / np.linalg.norm(major_profile)
        minor_profile = minor_profile / np.linalg.norm(minor_profile)
        chroma_norm = chroma_mean / np.linalg.norm(chroma_mean)
        
        # Tester toutes les tonalités
        best_correlation = -1
        best_key = 0
        best_mode = 'major'
        
        for i in range(12):
            # Rotation du chromagramme
            rotated = np.roll(chroma_norm, -i)
            
            # Corrélation avec majeur
            corr_major = np.corrcoef(rotated, major_profile)[0, 1]
            if corr_major > best_correlation:
                best_correlation = corr_major
                best_key = i
                best_mode = 'major'
            
            # Corrélation avec mineur
            corr_minor = np.corrcoef(rotated, minor_profile)[0, 1]
            if corr_minor > best_correlation:
                best_correlation = corr_minor
                best_key = i
                best_mode = 'minor'
        
        key_name = self.KEY_NAMES[best_key]
        
        return {
            'key': key_name,
            'mode': best_mode,
            'full_name': f"{key_name} {best_mode}",
            'confidence': round(best_correlation, 3),
            'key_index': best_key
        }
    
    def get_compatible_keys(self, key: str, mode: str = 'major') -> list[str]:
        """
        Retourne les tonalités compatibles pour une transition harmonique.
        
        Args:
            key: Tonalité (ex: 'C', 'F#')
            mode: Mode ('major' ou 'minor')
            
        Returns:
            list: Tonalités compatibles
        """
        key_idx = self.KEY_NAMES.index(key)
        
        # Tonalités compatibles :
        # - Même tonalité
        # - Quinte supérieure (+7 demi-tons)
        # - Quarte supérieure (+5 demi-tons)
        # - Relative majeure/mineure (+3 ou -3 demi-tons)
        
        compatible_offsets = [0, 7, 5, 3, -3]
        
        compatible = []
        for offset in compatible_offsets:
            comp_idx = (key_idx + offset) % 12
            compatible.append(self.KEY_NAMES[comp_idx])
        
        return compatible
    
    def check_compatibility(self, key1: str, key2: str) -> dict:
        """
        Vérifie la compatibilité entre deux tonalités.
        
        Args:
            key1: Première tonalité
            key2: Deuxième tonalité
            
        Returns:
            dict: Score de compatibilité et recommandation
        """
        compatible_keys = self.get_compatible_keys(key1)
        
        if key2 == key1:
            return {'score': 1.0, 'level': 'perfect', 'message': 'Même tonalité'}
        elif key2 in compatible_keys:
            return {'score': 0.8, 'level': 'good', 'message': 'Tonalités compatibles'}
        else:
            # Calculer la distance sur le cercle des quintes
            idx1 = self.KEY_NAMES.index(key1)
            idx2 = self.KEY_NAMES.index(key2)
            distance = min(abs(idx1 - idx2), 12 - abs(idx1 - idx2))
            score = max(0.3, 1.0 - (distance * 0.15))
            return {'score': round(score, 2), 'level': 'fair', 'message': f'Distance: {distance} demi-tons'}


def detect_key(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    """
    Fonction simple pour détecter la tonalité.
    
    Args:
        audio: Signal audio
        sample_rate: Fréquence d'échantillonnage
        
    Returns:
        dict: Informations sur la tonalité
    """
    analyzer = KeyAnalyzer(sample_rate)
    return analyzer.detect_key(audio)