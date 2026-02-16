"""Test de la génération de transition par IA."""

from src.transition.generator import TransitionGenerator

# Configuration
TRACK1 = "data/input/Far_from_Any_Road.mp3"
TRACK2 = "data/input/The_Fade_Out_Line.mp3"

generator = TransitionGenerator()

# Test avec le style smooth et chevauchement de 3 secondes
result = generator.generate_transition(
    TRACK1,
    TRACK2,
    transition_duration=15.0,    # 15 secondes de transition IA
    output_path="data/output/mix_final.wav",
    style='smooth',              # Style de transition
    overlap_duration=3.0         # 3 secondes de chevauchement de chaque côté
)

print(f"\n🎧 Écoutez le résultat: data/output/mix_final.wav")
print(f"\n📊 Résumé:")
print(f"  - Style: {result['style']}")
print(f"  - Point de transition morceau 1: {result['outro_time']:.2f}s")
print(f"  - Point de transition morceau 2: {result['intro_time']:.2f}s")
print(f"  - Durée transition IA: {result['transition_duration']:.2f}s")
print(f"  - Chevauchement: {result['overlap_duration']:.2f}s de chaque côté")
print(f"  - Durée totale: {result['duration']:.2f}s")