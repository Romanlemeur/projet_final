"""Entraînement intensif avec points de coupe."""

import torch
import torch.optim as optim
import os

from src.ai.transition_params_model import TransitionParamsVAE, vae_loss
from src.ai.params_dataset import TransitionParamsDataset


class ParamsVAETrainer:
    def __init__(self, model: TransitionParamsVAE):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        print(f"🖥️ Device: {self.device}")
        print(f"📊 Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    def train(self, dataset, epochs=500, batch_size=32, lr=0.0005, save_path="models/params_vae.pth"):
        print(f"\n🚀 Entraînement intensif")
        print(f"   Époques: {epochs}")
        print(f"   Batch: {batch_size}")
        print(f"   Dataset: {len(dataset)} échantillons")
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        
        best_loss = float('inf')
        n_batches = max(1, len(dataset) // batch_size)
        
        print(f"\n{'='*60}")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            for _ in range(n_batches):
                inputs, targets = dataset.get_batch(batch_size)
                inputs = torch.FloatTensor(inputs).to(self.device)
                targets = torch.FloatTensor(targets).to(self.device)
                
                optimizer.zero_grad()
                params_pred, mu, log_var = self.model(inputs)
                loss, recon, kl = vae_loss(params_pred, targets, mu, log_var)
                
                if torch.isnan(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
            avg_loss = epoch_loss / n_batches
            
            if (epoch + 1) % 25 == 0 or epoch == 0:
                lr_current = optimizer.param_groups[0]['lr']
                print(f"Époque {epoch+1:4d}/{epochs} | Loss: {avg_loss:.6f} | LR: {lr_current:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model(save_path)
        
        print(f"{'='*60}")
        print(f"\n✅ Terminé ! Loss: {best_loss:.6f}")
    
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_dim,
            'latent_dim': self.model.latent_dim,
            'output_dim': self.model.output_dim
        }, path)


def train_params_model(music_folder="data/input", epochs=500, max_samples=800,
                       batch_size=32, learning_rate=0.0005, save_path="models/params_vae.pth"):
    
    print("=" * 60)
    print("🧠 ENTRAÎNEMENT IA COMPLET (avec points de coupe)")
    print("=" * 60)
    
    dataset = TransitionParamsDataset()
    dataset.build_from_folder(music_folder, max_samples=max_samples)
    
    if len(dataset) == 0:
        return
    
    # Triple augmentation
    dataset.augment()
    dataset.augment()
    dataset.augment()
    
    print(f"\n   Dataset final: {len(dataset)} échantillons")
    dataset.save("data/dataset/params_dataset.pkl")
    
    # Modèle avec nouvelles dimensions
    model = TransitionParamsVAE(
        input_dim=28,      # 14 features x 2 morceaux
        hidden_dim=512,
        latent_dim=128,
        output_dim=38      # 38 paramètres (incluant points de coupe)
    )
    
    trainer = ParamsVAETrainer(model)
    trainer.train(dataset, epochs=epochs, batch_size=batch_size, lr=learning_rate, save_path=save_path)