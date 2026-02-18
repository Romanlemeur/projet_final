import torch
import torch.optim as optim
import os

from src.ai.transition_params_model import TransitionParamsVAE, vae_loss
from src.ai.params_dataset import TransitionParamsDataset


class ParamsVAETrainer:
    def __init__(self, model):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        print(f"Device: {self.device}")

    def train(self, dataset, epochs=500, batch_size=64, lr=0.0003, save_path="models/params_vae.pth"):
        print(f"Entrainement: {epochs} epoques, {len(dataset)} echantillons")
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        
        best_loss = float('inf')
        n_batches = max(1, len(dataset) // batch_size)
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            for _ in range(n_batches):
                inputs, targets = dataset.get_batch(batch_size)
                inputs = torch.FloatTensor(inputs).to(self.device)
                targets = torch.FloatTensor(targets).to(self.device)
                
                optimizer.zero_grad()
                params_pred, mu, log_var = self.model(inputs)
                loss, _, _ = vae_loss(params_pred, targets, mu, log_var)
                
                if torch.isnan(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
            avg_loss = epoch_loss / n_batches
            
            if (epoch + 1) % 50 == 0:
                print(f"Epoque {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model(save_path)
        
        print(f"Termine. Best loss: {best_loss:.6f}")

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_dim,
            'latent_dim': self.model.latent_dim,
            'output_dim': self.model.output_dim
        }, path)


def train_params_model(music_folder="data/input/music_train/fma_small/fma_small",
                       epochs=500,
                       max_songs=2000,
                       max_pairs=10000,
                       batch_size=64,
                       learning_rate=0.0003,
                       save_path="models/params_vae.pth"):
    
    print("=" * 50)
    print("ENTRAINEMENT IA DJ - 24 PARAMETRES")
    print("=" * 50)
    
    dataset = TransitionParamsDataset()
    dataset.build_from_folder(music_folder, max_songs=max_songs, max_pairs=max_pairs)
    
    if len(dataset) == 0:
        print("Erreur: aucun echantillon")
        return
    
    dataset.save("data/dataset/params_dataset.pkl")
    
    model = TransitionParamsVAE(
        input_dim=28,
        hidden_dim=512,
        latent_dim=128,
        output_dim=24
    )
    
    trainer = ParamsVAETrainer(model)
    trainer.train(dataset, epochs=epochs, batch_size=batch_size, lr=learning_rate, save_path=save_path)
