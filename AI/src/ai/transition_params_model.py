import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransitionParamsEncoder(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=512, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(hidden_dim // 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 4, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class TransitionParamsDecoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=512, output_dim=24):
        super().__init__()
        self.net = nn.Sequential(
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
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)


class TransitionParamsVAE(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=512, latent_dim=128, output_dim=24):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.encoder = TransitionParamsEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = TransitionParamsDecoder(latent_dim, hidden_dim, output_dim)
        
        self.param_names = [
            'low_eq_1', 'mid_eq_1', 'high_eq_1',
            'low_eq_2', 'mid_eq_2', 'high_eq_2',
            'volume_curve_1', 'volume_curve_2',
            'crossfade_type', 'crossfade_position',
            'cue_out_position', 'cue_in_position',
            'align_to_beat', 'align_to_bar',
            'transition_beats', 'eq_swap_timing', 'bass_swap_beat',
            'mix_style', 'filter_sweep', 'filter_resonance', 'tension_effect',
            'duck_vocals_1', 'duck_vocals_2', 'energy_direction'
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
        p = params_tensor.cpu().numpy()
        if p.ndim == 2:
            p = p[0]
        
        return {
            'low_eq_1': p[0] * 1.5 - 1.0,
            'mid_eq_1': p[1] * 1.5 - 1.0,
            'high_eq_1': p[2] * 1.5 - 1.0,
            'low_eq_2': p[3] * 1.5 - 1.0,
            'mid_eq_2': p[4] * 1.5 - 1.0,
            'high_eq_2': p[5] * 1.5 - 1.0,
            'volume_curve_1': p[6],
            'volume_curve_2': p[7],
            'crossfade_type': int(p[8] * 3),
            'crossfade_position': 0.3 + p[9] * 0.4,
            'cue_out_position': 0.6 + p[10] * 0.35,
            'cue_in_position': p[11] * 0.2,
            'align_to_beat': p[12],
            'align_to_bar': p[13],
            'transition_beats': int(16 + p[14] * 48),
            'eq_swap_timing': 0.3 + p[15] * 0.4,
            'bass_swap_beat': 0.4 + p[16] * 0.3,
            'mix_style': int(p[17] * 4),
            'filter_sweep': p[18],
            'filter_resonance': 0.1 + p[19] * 0.6,
            'tension_effect': int(p[20] * 4),
            'duck_vocals_1': p[21],
            'duck_vocals_2': p[22],
            'energy_direction': p[23]
        }


def vae_loss(params_pred, params_target, mu, log_var, kl_weight=0.0001):
    recon_loss = F.mse_loss(params_pred, params_target, reduction='mean')
    log_var = torch.clamp(log_var, min=-10, max=10)
    mu = torch.clamp(mu, min=-10, max=10)
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss
