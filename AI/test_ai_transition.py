"""Test de la transition DJ."""

from src.transition.generator import TransitionGenerator

TRACK1 = "data/input/The_Fade_Out_Line.mp3" 
TRACK2 = "data/input/White_Noise.mp3"              

generator = TransitionGenerator()

result = generator.generate_transition(
    TRACK1,
    TRACK2,
    transition_duration=24.0,  
    output_path="data/output/mix_transition_pro.wav",
    style='smooth',
    overlap_duration=5.0        
)

print(f"\n🎧 Écoutez: data/output/mix_transition_pro.wav")
print(f"   Durée totale: {result['duration']:.2f}s")