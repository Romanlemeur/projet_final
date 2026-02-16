"""Modèle VAE pour la génération de transitions musicales."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class Encoder(nn.Module):
    """Encodeur du VAE - compresse les spectrogrammes en espace latent."""
    
    def __init__(self, n_mels: int = 128, n_frames: int = 216, latent_dim: int = 128):
        super(Encoder, self).__init__()
        
        self.n_mels = n_mels
        self.n_frames = n_frames
        
        # Couches convolutionnelles pour chaque entrée
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        
        # Calculer la taille après convolutions
        self.conv_out_size = self._get_conv_output_size()
        
        # Couche de fusion des deux entrées
        self.fusion = nn.Sequential(
            nn.Linear(self.conv_out_size * 2, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
        )
        
        # Couches pour mu et log_var (VAE)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
    
    def _get_conv_output_size(self):
        """Calcule la taille de sortie des convolutions."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_mels, self.n_frames)
            out = self.conv1(dummy)
            return out.view(1, -1).size(1)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Encode deux spectrogrammes en espace latent.
        
        Args:
            x1: Spectrogramme fin morceau 1 (batch, 1, n_mels, n_frames)
            x2: Spectrogramme début morceau 2 (batch, 1, n_mels, n_frames)
            
        Returns:
            Tuple: (mu, log_var) paramètres de la distribution latente
        """
        # Encoder chaque entrée
        h1 = self.conv1(x1)
        h2 = self.conv1(x2)
        
        # Aplatir
        h1 = h1.view(h1.size(0), -1)
        h2 = h2.view(h2.size(0), -1)
        
        # Fusionner
        h = torch.cat([h1, h2], dim=1)
        h = self.fusion(h)
        
        # Paramètres latents
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        
        return mu, log_var


class Decoder(nn.Module):
    """Décodeur du VAE - génère le spectrogramme de transition."""
    
    def __init__(self, n_mels: int = 128, n_frames: int = 216, latent_dim: int = 128):
        super(Decoder, self).__init__()
        
        self.n_mels = n_mels
        self.n_frames = n_frames
        
        # Calculer les dimensions pour les convolutions transposées
        self.init_h = n_mels // 8
        self.init_w = n_frames // 8
        
        # Couche linéaire pour passer de latent à feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128 * self.init_h * self.init_w),
            nn.LeakyReLU(0.2),
        )
        
        # Convolutions transposées pour upsampling
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Sortie entre 0 et 1
        )
    
    def forward(self, z: torch.Tensor):
        """
        Décode l'espace latent en spectrogramme.
        
        Args:
            z: Vecteur latent (batch, latent_dim)
            
        Returns:
            Spectrogramme généré (batch, 1, n_mels, n_frames)
        """
        h = self.fc(z)
        h = h.view(h.size(0), 128, self.init_h, self.init_w)
        out = self.deconv(h)
        
        # Ajuster la taille si nécessaire
        out = F.interpolate(out, size=(self.n_mels, self.n_frames), mode='bilinear', align_corners=False)
        
        return out


class TransitionVAE(nn.Module):
    """
    VAE complet pour la génération de transitions.
    
    Le modèle prend deux spectrogrammes (fin morceau 1, début morceau 2)
    et génère un spectrogramme de transition.
    """
    
    def __init__(self, n_mels: int = 128, n_frames: int = 216, latent_dim: int = 128):
        super(TransitionVAE, self).__init__()
        
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(n_mels, n_frames, latent_dim)
        self.decoder = Decoder(n_mels, n_frames, latent_dim)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick pour le VAE.
        z = mu + std * epsilon
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Forward pass complet.
        
        Args:
            x1: Spectrogramme fin morceau 1
            x2: Spectrogramme début morceau 2
            
        Returns:
            Tuple: (reconstruction, mu, log_var)
        """
        # Encoder
        mu, log_var = self.encoder(x1, x2)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decoder
        reconstruction = self.decoder(z)
        
        return reconstruction, mu, log_var
    
    def generate(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Génère une transition (mode inférence).
        
        Args:
            x1: Spectrogramme fin morceau 1
            x2: Spectrogramme début morceau 2
            
        Returns:
            Spectrogramme de transition généré
        """
        self.eval()
        with torch.no_grad():
            mu, log_var = self.encoder(x1, x2)
            # En inférence, on utilise mu directement (pas de sampling)
            z = mu
            transition = self.decoder(z)
        return transition
    
    def generate_with_variation(self, x1: torch.Tensor, x2: torch.Tensor, 
                                 temperature: float = 1.0) -> torch.Tensor:
        """
        Génère une transition avec variation aléatoire.
        
        Args:
            x1, x2: Spectrogrammes d'entrée
            temperature: Contrôle la variation (0 = déterministe, >1 = plus de variation)
            
        Returns:
            Spectrogramme de transition
        """
        self.eval()
        with torch.no_grad():
            mu, log_var = self.encoder(x1, x2)
            std = torch.exp(0.5 * log_var) * temperature
            eps = torch.randn_like(std)
            z = mu + eps * std
            transition = self.decoder(z)
        return transition


def vae_loss(reconstruction: torch.Tensor, target: torch.Tensor, 
             mu: torch.Tensor, log_var: torch.Tensor,
             kl_weight: float = 0.0001) -> torch.Tensor:  # Réduit de 0.001 à 0.0001
    """
    Fonction de perte du VAE.
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstruction, target, reduction='mean')
    
    # KL Divergence avec clamp pour éviter l'explosion
    log_var = torch.clamp(log_var, min=-10, max=10)  # AJOUTER CETTE LIGNE
    mu = torch.clamp(mu, min=-10, max=10)            # AJOUTER CETTE LIGNE
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Loss totale
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss