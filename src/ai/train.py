"""Script d'entraînement du modèle VAE."""

import torch
import torch.optim as optim
import numpy as np
import os
from datetime import datetime

from src.ai.dataset import TransitionDataset
from src.ai.vae_model import TransitionVAE, vae_loss


class VAETrainer:
    """Entraîne le modèle VAE de transition."""
    
    def __init__(self, model: TransitionVAE, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        print(f"🖥️ Device: {self.device}")
        print(f"📊 Paramètres du modèle: {sum(p.numel() for p in model.parameters()):,}")
    
    def train(self, dataset: TransitionDataset, 
              epochs: int = 100,
              batch_size: int = 8,
              learning_rate: float = 0.001,
              save_path: str = "models/transition_vae.pth"):
        """
        Entraîne le modèle.
        
        Args:
            dataset: Dataset d'entraînement
            epochs: Nombre d'époques
            batch_size: Taille des batches
            learning_rate: Taux d'apprentissage
            save_path: Chemin pour sauvegarder le modèle
        """
        print(f"\n🚀 Démarrage de l'entraînement")
        print(f"  - Époques: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Dataset: {len(dataset)} échantillons")
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_loss = float('inf')
        history = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}
        
        # Calculer le nombre de batches par époque
        n_batches = max(1, len(dataset) // batch_size)
        
        print(f"\n{'='*60}")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = {'total': 0, 'recon': 0, 'kl': 0}
            
            for batch_idx in range(n_batches):
                # Récupérer un batch
                input1, input2, target = dataset.get_batch(batch_size)
                
                # Convertir en tenseurs
                input1 = torch.FloatTensor(input1).unsqueeze(1).to(self.device)
                input2 = torch.FloatTensor(input2).unsqueeze(1).to(self.device)
                target = torch.FloatTensor(target).unsqueeze(1).to(self.device)
                
                # Forward
                optimizer.zero_grad()
                reconstruction, mu, log_var = self.model(input1, input2)
                
                # Loss
                total_loss, recon_loss, kl_loss = vae_loss(
                    reconstruction, target, mu, log_var, kl_weight=0.001
                )
                
                # Backward
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Accumuler les pertes
                epoch_losses['total'] += total_loss.item()
                epoch_losses['recon'] += recon_loss.item()
                epoch_losses['kl'] += kl_loss.item()
            
            # Moyennes
            avg_total = epoch_losses['total'] / n_batches
            avg_recon = epoch_losses['recon'] / n_batches
            avg_kl = epoch_losses['kl'] / n_batches
            
            # Historique
            history['total_loss'].append(avg_total)
            history['recon_loss'].append(avg_recon)
            history['kl_loss'].append(avg_kl)
            
            # Scheduler
            scheduler.step(avg_total)
            
            # Affichage
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Époque {epoch+1:3d}/{epochs} | "
                      f"Loss: {avg_total:.6f} | "
                      f"Recon: {avg_recon:.6f} | "
                      f"KL: {avg_kl:.6f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Sauvegarder le meilleur modèle
            if avg_total < best_loss:
                best_loss = avg_total
                self.save_model(save_path)
        
        print(f"{'='*60}")
        print(f"\n✅ Entraînement terminé !")
        print(f"  - Meilleure loss: {best_loss:.6f}")
        print(f"  - Modèle sauvegardé: {save_path}")
        
        return history
    
    def save_model(self, path: str):
        """Sauvegarde le modèle."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_mels': self.model.n_mels,
            'n_frames': self.model.n_frames,
            'latent_dim': self.model.latent_dim
        }, path)
    
    def load_model(self, path: str):
        """Charge le modèle."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Modèle chargé: {path}")


def train_model(music_folder: str, 
                epochs: int = 100,
                batch_size: int = 8,
                max_samples: int = 100,
                save_path: str = "models/transition_vae.pth"):
    """
    Fonction utilitaire pour entraîner le modèle.
    
    Args:
        music_folder: Dossier contenant les musiques
        epochs: Nombre d'époques
        batch_size: Taille des batches
        max_samples: Nombre maximum d'échantillons
        save_path: Chemin de sauvegarde
    """
    print("="*60)
    print("🎵 ENTRAÎNEMENT DU MODÈLE VAE DE TRANSITION")
    print("="*60)
    
    # Créer le dataset
    print("\n📊 Création du dataset...")
    dataset = TransitionDataset(segment_duration=5.0)
    dataset.build_from_folder(music_folder, max_samples=max_samples)
    
    if len(dataset) == 0:
        print("❌ Erreur: Aucun échantillon créé !")
        return None
    
    # Augmenter les données
    dataset.augment_data()
    
    # Sauvegarder le dataset
    dataset.save("data/dataset/transition_dataset.pkl")
    
    # Obtenir les dimensions
    sample = dataset.data[0]
    n_mels = sample['input1'].shape[0]
    n_frames = sample['input1'].shape[1]
    
    print(f"\n📐 Dimensions: {n_mels} mels x {n_frames} frames")
    
    # Créer le modèle
    model = TransitionVAE(n_mels=n_mels, n_frames=n_frames, latent_dim=128)
    
    # Entraîner
    trainer = VAETrainer(model)
    history = trainer.train(
        dataset,
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path
    )
    
    return history  