"""Script pour entraîner le modèle VAE."""

from src.ai.train import train_model

# Entraîner le modèle avec vos musiques
history = train_model(
    music_folder="data/input",      # Dossier avec vos MP3
    epochs=100,                      # Nombre d'époques
    batch_size=4,                    # Taille des batches
    max_samples=50,                  # Nombre d'échantillons à créer
    save_path="models/transition_vae.pth"  # Où sauvegarder
)

print("\n🎉 Entraînement terminé !")
print("Maintenant testez avec: python test_ai_transition.py")