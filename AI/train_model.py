from src.ai.train_params import train_params_model

print("=" * 50)
print("ENTRAINEMENT IA DJ")
print("=" * 50)

train_params_model(
    music_folder="data/input/music_train/fma_small/fma_small",
    epochs=500,
    max_songs=2000,
    max_pairs=10000,
    batch_size=64,
    learning_rate=0.0003,
    save_path="models/params_vae.pth"
)

print("Termine.")
print("Test: python test_ai_transition.py")
