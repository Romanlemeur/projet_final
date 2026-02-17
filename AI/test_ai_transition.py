"""Test de la transition DJ."""

from src.transition.generator import TransitionGenerator

# Configuration
TRACK1 = "data/input/White_Noise.mp3"  # Changez selon vos fichiers
TRACK2 = "data/input/Far_from_Any_Road.mp3"              # Changez selon vos fichiers

generator = TransitionGenerator()

# Transition LONGUE (24 secondes)
result = generator.generate_transition(
    TRACK1,
    TRACK2,
    transition_duration=24.0,   # Plus long !
    output_path="data/output/mix_transition_pro.wav",
    style='smooth',
    overlap_duration=5.0        # Chevauchement plus long aussi
)

print(f"\n🎧 Écoutez: data/output/mix_transition_pro.wav")
print(f"   Durée totale: {result['duration']:.2f}s")