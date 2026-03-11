import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa

N_MELS = 64
HOP_LENGTH = 512
N_FFT = 2048
WINDOW_SECONDS = 5
ENCODER_DIM = 32
VAE_LATENT = 64
N_PARAMS = 24
TIME_FRAMES = 216


def compute_mel(audio, sample_rate, n_mels=N_MELS, hop_length=HOP_LENGTH,
                n_fft=N_FFT, window_seconds=WINDOW_SECONDS,
                time_frames=TIME_FRAMES):
    target_samples = int(window_seconds * sample_rate)
    if len(audio) >= target_samples:
        audio = audio[:target_samples]
    else:
        audio = np.pad(audio, (0, target_samples - len(audio)))

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=n_mels,
        hop_length=hop_length, n_fft=n_fft
    )
    mel_db = librosa.power_to_db(mel + 1e-10, ref=np.max)
    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max > mel_min:
        mel_db = (mel_db - mel_min) / (mel_max - mel_min)
    else:
        mel_db = np.zeros_like(mel_db)

    if mel_db.shape[1] != time_frames:
        indices = np.linspace(0, mel_db.shape[1] - 1, time_frames)
        left = np.floor(indices).astype(int)
        right = np.minimum(left + 1, mel_db.shape[1] - 1)
        frac = indices - left
        mel_db = mel_db[:, left] * (1 - frac) + mel_db[:, right] * frac

    return mel_db.astype(np.float32)


class MelEncoder(nn.Module):
    def __init__(self, n_mels=N_MELS, embedding_dim=ENCODER_DIM):
        super().__init__()
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128), nn.GELU(),
            nn.Linear(128, embedding_dim),
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.GELU(),
            nn.Linear(128, 64 * 4 * 4), nn.GELU(),
        )
        self.decoder_cnn = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid(),
        )

    def encode(self, mel):
        x = mel.unsqueeze(1)
        x = self.encoder_cnn(x)
        x = x.view(x.size(0), -1)
        return self.encoder_fc(x)

    def decode(self, z, target_h, target_w):
        x = self.decoder_fc(z)
        x = x.view(x.size(0), 64, 4, 4)
        x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        x = self.decoder_cnn(x)
        return x.squeeze(1)

    def forward(self, mel):
        z = self.encode(mel)
        recon = self.decode(z, mel.size(1), mel.size(2))
        return z, recon

    def embed_audio(self, audio, sample_rate, device='cpu'):
        self.eval()
        with torch.no_grad():
            mel = compute_mel(audio, sample_rate)
            t = torch.FloatTensor(mel).unsqueeze(0).to(device)
            emb = self.encode(t)
        return emb[0].cpu().numpy()


PARAM_NAMES = [
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


class MelTransitionVAE(nn.Module):
    def __init__(self, embedding_dim=ENCODER_DIM * 2, latent_dim=VAE_LATENT,
                 hidden_dim=256, output_dim=N_PARAMS):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.param_names = PARAM_NAMES

        self.enc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim), nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            h = self.enc(x)
            mu = self.fc_mu(h)
            return self.dec(mu)

    def get_params_dict(self, params_tensor):
        p = params_tensor.cpu().numpy()
        if p.ndim == 2:
            p = p[0]
        return {
            'low_eq_1':          p[0] * 1.5 - 1.0,
            'mid_eq_1':          p[1] * 1.5 - 1.0,
            'high_eq_1':         p[2] * 1.5 - 1.0,
            'low_eq_2':          p[3] * 1.5 - 1.0,
            'mid_eq_2':          p[4] * 1.5 - 1.0,
            'high_eq_2':         p[5] * 1.5 - 1.0,
            'volume_curve_1':    p[6],
            'volume_curve_2':    p[7],
            'crossfade_type':    int(p[8] * 3),
            'crossfade_position': 0.3 + p[9] * 0.4,
            'cue_out_position':  0.6 + p[10] * 0.35,
            'cue_in_position':   p[11] * 0.2,
            'align_to_beat':     p[12],
            'align_to_bar':      p[13],
            'transition_beats':  int(16 + p[14] * 48),
            'eq_swap_timing':    0.3 + p[15] * 0.4,
            'bass_swap_beat':    0.4 + p[16] * 0.3,
            'mix_style':         int(p[17] * 4),
            'filter_sweep':      p[18],
            'filter_resonance':  0.1 + p[19] * 0.6,
            'tension_effect':    int(p[20] * 4),
            'duck_vocals_1':     p[21],
            'duck_vocals_2':     p[22],
            'energy_direction':  p[23],
        }
