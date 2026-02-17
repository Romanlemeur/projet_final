"""Modèle VAE complet - inclut les points de coupe optimaux."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransitionParamsEncoder(nn.Module):
    """Encodeur profond."""
    
    def __init__(self, input_dim: int = 28, hidden_dim: int = 512, latent_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 4, latent_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)


class TransitionParamsDecoder(nn.Module):
    """Décodeur profond."""
    
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 512, output_dim: int = 38):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 4),
            
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 2),
            
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.decoder(z)


class TransitionParamsVAE(nn.Module):
    """
    VAE complet qui prédit TOUS les paramètres incluant les points de coupe.
    
    L'IA décide:
    - OÙ couper le morceau 1 (point de sortie optimal)
    - OÙ reprendre le morceau 2 (point d'entrée optimal)
    - Structure de la transition
    - Filtres et effets
    - Style et mix
    """
    
    def __init__(self, input_dim: int = 28, hidden_dim: int = 512, 
                 latent_dim: int = 128, output_dim: int = 38):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        self.encoder = TransitionParamsEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = TransitionParamsDecoder(latent_dim, hidden_dim, output_dim)
        
        self.param_names = [
            # === POINTS DE COUPE (8 params) - NOUVEAU ===
            'track1_cut_position',      # 0: Où couper track 1 (0=début, 1=fin)
            'track1_cut_on_beat',       # 1: Force de préférence pour couper sur un beat
            'track1_cut_on_bar',        # 2: Force de préférence pour couper sur une mesure
            'track1_fade_before_cut',   # 3: Durée du fade avant la coupe
            'track2_start_position',    # 4: Où commencer track 2 (0=début, 1=plus tard)
            'track2_start_on_beat',     # 5: Force de préférence pour commencer sur un beat
            'track2_start_on_bar',      # 6: Force de préférence pour commencer sur une mesure
            'track2_fade_after_start',  # 7: Durée du fade après le début
            
            # === STRUCTURE (5 params) ===
            'phase1_duration',          # 8
            'phase2_duration',          # 9
            'phase3_duration',          # 10
            'total_duration_factor',    # 11
            'overlap_duration',         # 12
            
            # === FILTRES PHASE 1 (4 params) ===
            'p1_filter_start',          # 13
            'p1_filter_end',            # 14
            'p1_filter_resonance',      # 15
            'p1_filter_curve',          # 16
            
            # === FILTRES PHASE 3 (4 params) ===
            'p3_filter_start',          # 17
            'p3_filter_end',            # 18
            'p3_filter_resonance',      # 19
            'p3_filter_curve',          # 20
            
            # === EFFETS (6 params) ===
            'reverb_amount',            # 21
            'reverb_decay',             # 22
            'echo_amount',              # 23
            'echo_delay',               # 24
            'echo_feedback',            # 25
            'use_riser',                # 26
            
            # === MIX (6 params) ===
            'drums_volume',             # 27
            'harmonic_volume',          # 28
            'bass_volume',              # 29
            'bass_swap_point',          # 30
            'crossfade_curve',          # 31
            'energy_curve',             # 32
            
            # === STYLE (5 params) ===
            'style_smooth',             # 33
            'style_dramatic',           # 34
            'style_ambient',            # 35
            'brightness_target',        # 36
            'tension_level',            # 37
        ]
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        params = self.decoder(z)
        return params, mu, log_var
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            mu, _ = self.encoder(x)
            params = self.decoder(mu)
        return params
    
    def get_params_dict(self, params_tensor):
        params = params_tensor.cpu().numpy()
        if params.ndim == 2:
            params = params[0]
        return {name: float(params[i]) for i, name in enumerate(self.param_names)}


def vae_loss(params_pred, params_target, mu, log_var, kl_weight=0.0001):
    recon_loss = F.mse_loss(params_pred, params_target, reduction='mean')
    
    log_var = torch.clamp(log_var, min=-10, max=10)
    mu = torch.clamp(mu, min=-10, max=10)
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss