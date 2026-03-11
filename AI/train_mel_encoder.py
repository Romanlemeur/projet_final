import argparse
import os
import sys
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
import librosa
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from src.ai.mel_encoder import MelEncoder, MelTransitionVAE, compute_mel, N_MELS, ENCODER_DIM, TIME_FRAMES
from src.utils.config import SAMPLE_RATE


def _cache_worker(args):
    path, cache_file, sample_rate, window_sec, offsets = args
    windows = []
    for offset in offsets:
        try:
            audio, _ = librosa.load(str(path), sr=sample_rate, mono=True,
                                     offset=offset, duration=window_sec)
            if len(audio) < sample_rate:
                continue
            mel = compute_mel(audio, sample_rate)
            windows.append(mel.astype(np.float16))
        except Exception:
            pass
    if windows:
        np.save(str(cache_file), np.array(windows, dtype=np.float16))
        return 'ok'
    return 'error'


def build_mel_cache(audio_files, cache_dir, sample_rate=SAMPLE_RATE,
                    window_sec=5.0, windows_per_file=2, n_workers=4):
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    if windows_per_file == 1:
        offsets = [3.0]
    else:
        span = 25.0 - window_sec
        offsets = [3.0 + i * span / max(windows_per_file - 1, 1)
                   for i in range(windows_per_file)]

    work = []
    already_cached = 0
    for path in audio_files:
        stem = Path(path).stem
        cache_file = cache_path / f"{stem}.npy"
        if cache_file.exists():
            already_cached += 1
        else:
            work.append((str(path), str(cache_file), sample_rate, window_sec, offsets))

    total = len(audio_files)
    print(f"\n  Fichiers à mettre en cache : {len(work):,}  |  Déjà en cache : {already_cached:,}  |  Workers : {n_workers}")

    if work:
        t0 = time.time()
        done = already_cached
        ok = 0
        errors = 0
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_cache_worker, w): w for w in work}
            for future in as_completed(futures):
                result = future.result()
                done += 1
                if result == 'ok':
                    ok += 1
                else:
                    errors += 1
                if done % 2000 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = ok / max(elapsed, 1)
                    eta = (len(work) - ok) / max(rate, 1)
                    print(f"  Cache : {done:,}/{total:,} | {rate:.1f} fichiers/s | ETA {eta/60:.0f} min | erreurs : {errors}")

        print(f"\n  Cache construit : {ok:,} nouveaux | {errors} erreurs | {time.time()-t0:.0f}s écoulées")

    npy_files = sorted(cache_path.glob('*.npy'))
    print(f"[OK] Cache prêt : {len(npy_files):,} fichiers dans '{cache_dir}'")
    return npy_files


class CachedMelDataset(Dataset):
    def __init__(self, npy_files, windows_per_file=2):
        self.items = [(str(f), i) for f in npy_files for i in range(windows_per_file)]
        print(f"  CachedMelDataset : {len(npy_files):,} fichiers × {windows_per_file} fenêtres "
              f"= {len(self.items):,} échantillons")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, win_idx = self.items[idx]
        try:
            arr = np.load(path)
            win_idx = min(win_idx, len(arr) - 1)
            mel = arr[win_idx].astype(np.float32)
        except Exception:
            mel = np.zeros((N_MELS, TIME_FRAMES), dtype=np.float32)
        return torch.FloatTensor(mel)


class MelWindowDataset(Dataset):
    def __init__(self, audio_files, sample_rate=SAMPLE_RATE, window_sec=5.0,
                 windows_per_file=2):
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.window_samples = int(window_sec * sample_rate)

        if windows_per_file == 1:
            offsets = [3.0]
        else:
            span = 25.0 - window_sec
            offsets = [3.0 + i * span / max(windows_per_file - 1, 1)
                       for i in range(windows_per_file)]

        self.items = [(str(p), off) for p in audio_files for off in offsets]
        print(f"  LazyMelDataset : {len(audio_files)} fichiers × {len(offsets)} = {len(self.items)} échantillons")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, offset_sec = self.items[idx]
        try:
            audio, _ = librosa.load(path, sr=self.sample_rate, mono=True,
                                     offset=offset_sec, duration=self.window_sec)
            if len(audio) < self.sample_rate:
                audio = np.zeros(self.window_samples, dtype=np.float32)
        except Exception:
            audio = np.zeros(self.window_samples, dtype=np.float32)
        mel = compute_mel(audio, self.sample_rate)
        return torch.FloatTensor(mel)


GENRE_PROFILES = {
    'electronic': {'bpm': 130, 'energy': 0.75, 'low': 0.45, 'mid': 0.35, 'high': 0.20,
                   'beat_str': 0.90, 'vocal': 0.10, 'breakdown': 0.55},
    'techno':     {'bpm': 140, 'energy': 0.80, 'low': 0.42, 'mid': 0.33, 'high': 0.25,
                   'beat_str': 0.95, 'vocal': 0.05, 'breakdown': 0.60},
    'house':      {'bpm': 125, 'energy': 0.70, 'low': 0.40, 'mid': 0.38, 'high': 0.22,
                   'beat_str': 0.85, 'vocal': 0.35, 'breakdown': 0.50},
    'pop':        {'bpm': 115, 'energy': 0.62, 'low': 0.30, 'mid': 0.42, 'high': 0.28,
                   'beat_str': 0.65, 'vocal': 0.70, 'breakdown': 0.35},
    'hiphop':     {'bpm': 95,  'energy': 0.68, 'low': 0.50, 'mid': 0.35, 'high': 0.15,
                   'beat_str': 0.80, 'vocal': 0.60, 'breakdown': 0.40},
    'ambient':    {'bpm': 80,  'energy': 0.30, 'low': 0.25, 'mid': 0.50, 'high': 0.25,
                   'beat_str': 0.20, 'vocal': 0.10, 'breakdown': 0.75},
    'rock':       {'bpm': 120, 'energy': 0.72, 'low': 0.32, 'mid': 0.40, 'high': 0.28,
                   'beat_str': 0.75, 'vocal': 0.55, 'breakdown': 0.30},
    'rnb':        {'bpm': 90,  'energy': 0.58, 'low': 0.38, 'mid': 0.40, 'high': 0.22,
                   'beat_str': 0.65, 'vocal': 0.65, 'breakdown': 0.42},
}


def _generate_synthetic_mel(profile, n_mels=N_MELS, time_frames=TIME_FRAMES, variation=0.25):
    rng = np.random.default_rng()

    p = {
        k: float(np.clip(v + rng.normal(0, variation * v if isinstance(v, float) else 0), 0.01, 1.0))
        if isinstance(v, float) else v
        for k, v in profile.items()
    }
    bpm = profile['bpm'] * (1 + rng.uniform(-0.08, 0.08))

    low_end = int(n_mels * 0.25)
    mid_end = int(n_mels * 0.65)

    mel = np.zeros((n_mels, time_frames), dtype=np.float32)
    mel[:low_end]        = rng.exponential(p['low'] * 0.6 + 0.05,  (low_end, time_frames))
    mel[low_end:mid_end] = rng.exponential(p['mid'] * 0.35 + 0.03, (mid_end - low_end, time_frames))
    mel[mid_end:]        = rng.exponential(p['high'] * 0.20 + 0.01, (n_mels - mid_end, time_frames))

    if p['beat_str'] > 0.3:
        beat_period = max(3, int(time_frames * 60 / bpm / 5))
        for t in range(0, time_frames, beat_period):
            mel[:low_end, t:min(t + 3, time_frames)] *= (1 + p['beat_str'] * 2.5)

    if p['vocal'] > 0.3:
        mel[int(n_mels * 0.35):int(n_mels * 0.75)] *= (1 + p['vocal'] * 0.8)

    mel *= p['energy']
    mel += rng.standard_normal((n_mels, time_frames)).astype(np.float32) * 0.01

    m, M = mel.min(), mel.max()
    if M > m:
        mel = (mel - m) / (M - m)
    return mel


class SyntheticMelDataset(Dataset):
    def __init__(self, n_samples=2000, n_mels=N_MELS, time_frames=TIME_FRAMES):
        self.n_samples = n_samples
        self.n_mels = n_mels
        self.time_frames = time_frames
        self.profiles = list(GENRE_PROFILES.values())

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        profile = self.profiles[idx % len(self.profiles)]
        mel = _generate_synthetic_mel(profile, self.n_mels, self.time_frames)
        return torch.FloatTensor(mel)


def _features_from_profile(profile, key_pos=None, mode_major=None):
    rng = np.random.default_rng()
    bpm    = profile['bpm'] * (1 + rng.uniform(-0.1, 0.1))
    energy = float(np.clip(profile['energy'] + rng.normal(0, 0.08), 0.05, 1.0))
    kp     = key_pos if key_pos is not None else rng.uniform(0, 1)
    mm     = mode_major if mode_major is not None else rng.choice([0.0, 1.0])
    low    = float(np.clip(profile['low']  + rng.normal(0, 0.05), 0.01, 1.0))
    mid    = float(np.clip(profile['mid']  + rng.normal(0, 0.05), 0.01, 1.0))
    high   = float(np.clip(profile['high'] + rng.normal(0, 0.03), 0.01, 1.0))
    total  = low + mid + high + 1e-8

    return np.array([
        bpm / 200.0, energy, kp, mm, rng.uniform(0.5, 1.0),
        low / total, mid / total, high / total,
        rng.uniform(0.1, 0.8), rng.uniform(0.2, 0.9), rng.uniform(0.05, 0.5),
        float(np.clip(profile['beat_str'] + rng.normal(0, 0.1), 0, 1)),
        rng.uniform(0.1, 0.7), rng.uniform(0.05, 0.5),
    ], dtype=np.float32), {
        'bpm': bpm, 'energy': energy, 'low_ratio': low / total,
        'mid_ratio': mid / total, 'high_ratio': high / total,
        'beat_regularity': profile['beat_str'], 'vocal_presence': profile['vocal'],
        'breakdown_score': profile['breakdown'],
        'camelot': '8B', 'key': 'C', 'mode': 'major',
    }


def _compute_ideal_params(f1, f2):
    from src.ai.params_dataset import TransitionParamsDataset as _DS
    _ds = _DS.__new__(_DS)
    _ds.circle_of_fifths = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']
    f1.setdefault('camelot', '8B')
    f2.setdefault('camelot', '8B')
    return _DS._compute_ideal_params(_ds, f1, f2)


class MelVAEDataset(Dataset):
    def __init__(self, encoder, device, n_samples=8000):
        self.encoder = encoder
        self.device = device
        self.profiles = list(GENRE_PROFILES.values())
        self.data = self._generate(n_samples)

    def _generate(self, n_samples):
        print(f"  Génération de {n_samples:,} paires d'embeddings...")
        data = []
        self.encoder.eval()
        with torch.no_grad():
            for i in range(n_samples):
                p1 = self.profiles[i % len(self.profiles)]
                p2 = self.profiles[(i + 3) % len(self.profiles)]

                mel1 = torch.FloatTensor(_generate_synthetic_mel(p1)).unsqueeze(0).to(self.device)
                mel2 = torch.FloatTensor(_generate_synthetic_mel(p2)).unsqueeze(0).to(self.device)

                emb1 = self.encoder.encode(mel1)[0].cpu().numpy()
                emb2 = self.encoder.encode(mel2)[0].cpu().numpy()

                _, feat1 = _features_from_profile(p1)
                _, feat2 = _features_from_profile(p2)
                target = _compute_ideal_params(feat1, feat2)

                data.append({
                    'input': np.concatenate([emb1, emb2]).astype(np.float32),
                    'target': target.astype(np.float32),
                })

                if (i + 1) % 1000 == 0:
                    print(f"    {i + 1:,}/{n_samples:,}")

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return torch.FloatTensor(d['input']), torch.FloatTensor(d['target'])


def train_mel_encoder(audio_files, output_path='models/mel_encoder.pth',
                      cache_dir=None, windows_per_file=2,
                      epochs=20, batch_size=64, lr=3e-4, cache_workers=4):
    print("\n" + "=" * 60)
    print("  PHASE 1 : ENCODEUR MEL  (auto-supervisé)")
    print("=" * 60)
    print(f"\nFichiers audio : {len(audio_files):,}")

    if cache_dir:
        npy_files = build_mel_cache(audio_files, cache_dir,
                                     windows_per_file=windows_per_file,
                                     n_workers=cache_workers)
        real_ds = CachedMelDataset(npy_files, windows_per_file=windows_per_file)
    else:
        real_ds = MelWindowDataset(audio_files, windows_per_file=windows_per_file)

    n_real = len(real_ds)
    n_synth = max(500, min(2000, n_real // 20))
    print(f"Ajout de {n_synth} échantillons mel synthétiques pour la régularisation...")
    synth_ds = SyntheticMelDataset(n_samples=n_synth)

    dataset = torch.utils.data.ConcatDataset([real_ds, synth_ds])
    print(f"Total d'échantillons d'entraînement : {len(dataset):,}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=0, pin_memory=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MelEncoder(n_mels=N_MELS, embedding_dim=ENCODER_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Appareil : {device} | Paramètres : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Entraînement {epochs} époques, batch {batch_size}, {len(loader):,} batches/époque\n")

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total = 0.0
        n = 0
        t0 = time.time()
        for mel_batch in loader:
            mel_batch = mel_batch.to(device)
            optimizer.zero_grad()
            z, recon = model(mel_batch)
            loss = nn.functional.mse_loss(recon, mel_batch)
            loss += 0.005 * torch.mean(z.pow(2))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            n += 1
        scheduler.step()
        avg = total / max(n, 1)
        elapsed = time.time() - t0
        if avg < best_loss:
            best_loss = avg
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save({
                'model_state': model.state_dict(),
                'embedding_dim': ENCODER_DIM,
                'n_mels': N_MELS,
                'loss': best_loss,
            }, output_path)
            saved = ' [sauvegardé]'
        else:
            saved = ''
        print(f"  Époque {epoch+1:3d}/{epochs} | Perte : {avg:.6f} | Meilleure : {best_loss:.6f} | {elapsed:.0f}s{saved}")

    print(f"\n[OK] MelEncoder entraîné. Meilleure perte : {best_loss:.6f}")
    print(f"[OK] Sauvegardé : {output_path}")
    return model.cpu(), best_loss


def train_mel_vae(encoder, output_path='models/mel_vae.pth',
                  n_samples=20000, epochs=80, batch_size=64, lr=3e-4):
    print("\n" + "=" * 60)
    print("  PHASE 2 : MEL TRANSITION VAE  (basé sur les embeddings audio)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = encoder.to(device)
    encoder.eval()

    print(f"\nGénération de {n_samples:,} paires d'entraînement à partir des profils de genre...")
    dataset = MelVAEDataset(encoder, device, n_samples=n_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = MelTransitionVAE(
        embedding_dim=ENCODER_DIM * 2,
        latent_dim=64,
        hidden_dim=256,
        output_dim=24,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Appareil : {device} | Paramètres : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Entraînement {epochs} époques\n")

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for inp, target in loader:
            inp, target = inp.to(device), target.to(device)
            optimizer.zero_grad()
            pred, mu, logvar = model(inp)
            recon_loss = nn.functional.mse_loss(pred, target)
            logvar = torch.clamp(logvar, -10, 10)
            mu = torch.clamp(mu, -10, 10)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.0001 * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += recon_loss.item()
            n += 1
        scheduler.step()
        avg = total / max(n, 1)
        if avg < best_loss:
            best_loss = avg
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save({
                'model_state': model.state_dict(),
                'embedding_dim': ENCODER_DIM * 2,
                'latent_dim': 64,
                'output_dim': 24,
                'loss': best_loss,
            }, output_path)
            saved = ' [sauvegardé]'
        else:
            saved = ''
        if (epoch + 1) % 5 == 0:
            print(f"  Époque {epoch+1:3d}/{epochs} | Perte : {avg:.6f} | Meilleure : {best_loss:.6f}{saved}")

    print(f"\n[OK] MelTransitionVAE entraîné. Meilleure perte : {best_loss:.6f}")
    print(f"[OK] Sauvegardé : {output_path}")
    return best_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entraîner MelEncoder + MelTransitionVAE')
    parser.add_argument('--audio-folder',   default='data/input',
                        help='Dossier contenant les fichiers audio (recherche récursive). '
                             'Ex. data/fma_large')
    parser.add_argument('--cache-dir',      default='data/mel_cache',
                        help='Où stocker les fichiers mel .npy pré-calculés. '
                             'Passer une chaîne vide "" pour désactiver le cache (chargement lazy, lent sur de grands datasets).')
    parser.add_argument('--windows-per-file', type=int, default=2,
                        help='Fenêtres mel extraites par fichier audio (défaut : 2)')
    parser.add_argument('--cache-workers',  type=int, default=4,
                        help='Threads parallèles pour construire le cache mel (défaut : 4)')
    parser.add_argument('--encoder-output', default='models/mel_encoder.pth')
    parser.add_argument('--vae-output',     default='models/mel_vae.pth')
    parser.add_argument('--encoder-epochs', type=int, default=20,
                        help='Époques phase 1 (défaut 20 ; moins nécessaire avec un grand dataset)')
    parser.add_argument('--vae-epochs',     type=int, default=80)
    parser.add_argument('--batch-size',     type=int, default=64)
    parser.add_argument('--vae-samples',    type=int, default=20000,
                        help='Paires d\'entraînement synthétiques pour MelVAE (défaut : 20000)')
    parser.add_argument('--max-files',      type=int, default=0,
                        help='Limiter le nombre de fichiers audio (0 = pas de limite)')
    parser.add_argument('--skip-encoder',   action='store_true',
                        help='Ignorer la phase 1 ; charger les poids existants de l\'encodeur')
    parser.add_argument('--skip-vae',       action='store_true',
                        help='Ignorer la phase 2')
    args = parser.parse_args()

    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.flac', '*.ogg']:
        audio_files.extend(Path(args.audio_folder).rglob(ext))
    audio_files = sorted(audio_files)

    if args.max_files and len(audio_files) > args.max_files:
        random.shuffle(audio_files)
        audio_files = audio_files[:args.max_files]
        print(f"Limité à {args.max_files:,} fichiers (--max-files)")

    print(f"\n{len(audio_files):,} fichiers audio trouvés dans '{args.audio_folder}'")

    if not audio_files:
        print("[ERREUR] Aucun fichier audio trouvé. Vérifiez le chemin --audio-folder.")
        sys.exit(1)

    cache_dir = args.cache_dir if (args.cache_dir and len(audio_files) > 50) else None
    if cache_dir is None and args.cache_dir:
        print("(Petit dataset — cache ignoré, chargement lazy)")

    if not args.skip_encoder:
        encoder, _ = train_mel_encoder(
            audio_files=audio_files,
            output_path=args.encoder_output,
            cache_dir=cache_dir,
            windows_per_file=args.windows_per_file,
            epochs=args.encoder_epochs,
            batch_size=args.batch_size,
            cache_workers=args.cache_workers,
        )
    else:
        print("\n[IGNORÉ] Phase 1 : chargement de l'encodeur existant...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        encoder = MelEncoder(n_mels=N_MELS, embedding_dim=ENCODER_DIM)
        ckpt = torch.load(args.encoder_output, map_location='cpu', weights_only=True)
        encoder.load_state_dict(ckpt['model_state'])
        print(f"[OK] Encodeur chargé depuis {args.encoder_output}")

    if not args.skip_vae:
        train_mel_vae(
            encoder=encoder,
            output_path=args.vae_output,
            n_samples=args.vae_samples,
            epochs=args.vae_epochs,
            batch_size=args.batch_size,
        )
