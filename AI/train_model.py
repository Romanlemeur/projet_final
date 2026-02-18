from src.ai.train_params import train_params_model

print("=" * 60)
print("  DJ AI TRANSITION - PROFESSIONAL TRAINING")
print("=" * 60)
print()
print("  Features:")
print("    - Vocal detection (MFCC + Spectral analysis)")
print("    - Structure analysis (Intro/Breakdown/Drop/Outro)")
print("    - Harmonic mixing (Camelot Wheel)")
print("    - Phrase detection (4/8/16/32 bars)")
print("    - Loop quality scoring")
print()

train_params_model(
    music_folder="data/input/music_train/fma_small/fma_small",
    epochs=500,
    max_songs=200,
    max_pairs=2000,
    batch_size=32,
    learning_rate=0.0003,
    save_path="models/params_vae.pth"
)

print()
print("  Next step: python test_ai_transition.py")
print()
