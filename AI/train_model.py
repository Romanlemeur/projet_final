"""Entraîner le modèle IA complet avec points de coupe."""

from src.ai.train_params import train_params_model

print("=" * 60)
print("🧠 ENTRAÎNEMENT IA COMPLET")
print("   - 38 paramètres")
print("   - Points de coupe intelligents")
print("   - 500 époques")
print("=" * 60)
print("\n⚠️ Temps estimé: 10-30 minutes\n")

train_params_model(
    music_folder="data/input",
    epochs=500,
    max_samples=800,
    batch_size=32,
    learning_rate=0.0005,
    save_path="models/params_vae.pth"
)

print("\n🎉 IA entraînée avec succès !")
print("Testez: python test_ai_transition.py")