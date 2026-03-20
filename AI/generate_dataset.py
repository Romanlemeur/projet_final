import numpy as np
import pickle
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ai.params_dataset import TransitionParamsDataset

CAMELOT_KEYS = [
    '1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B',
    '5A', '5B', '6A', '6B', '7A', '7B', '8A', '8B',
    '9A', '9B', '10A', '10B', '11A', '11B', '12A', '12B'
]

CAMELOT_TO_INFO = {
    '8B':  ('C',   'major'), '8A':  ('A',   'minor'),
    '9B':  ('G',   'major'), '9A':  ('E',   'minor'),
    '10B': ('D',   'major'), '10A': ('B',   'minor'),
    '11B': ('A',   'major'), '11A': ('F#',  'minor'),
    '12B': ('E',   'major'), '12A': ('C#',  'minor'),
    '1B':  ('B',   'major'), '1A':  ('G#',  'minor'),
    '2B':  ('F#',  'major'), '2A':  ('D#',  'minor'),
    '3B':  ('Db',  'major'), '3A':  ('Bb',  'minor'),
    '4B':  ('Ab',  'major'), '4A':  ('F',   'minor'),
    '5B':  ('Eb',  'major'), '5A':  ('C',   'minor'),
    '6B':  ('Bb',  'major'), '6A':  ('G',   'minor'),
    '7B':  ('F',   'major'), '7A':  ('D',   'minor'),
}

CIRCLE_OF_FIFTHS = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']

GENRES = {
    'electronic': dict(
        bpm=(120, 150), energy=(0.6, 1.0), beat_regularity=(0.80, 1.0),
        vocal_presence=(0.0, 0.25), breakdown_score=(0.35, 0.85),
        low_ratio=(0.30, 0.50), mid_ratio=(0.20, 0.38),
        spectral_centroid=(0.25, 0.70), spectral_rolloff=(0.30, 0.75),
        spectral_flatness=(0.0, 0.20), rms_var=(0.05, 0.30),
        onset_density=(0.45, 0.90), key_confidence=(0.50, 0.95),
    ),
    'techno': dict(
        bpm=(132, 162), energy=(0.72, 1.0), beat_regularity=(0.85, 1.0),
        vocal_presence=(0.0, 0.12), breakdown_score=(0.40, 0.90),
        low_ratio=(0.35, 0.55), mid_ratio=(0.18, 0.33),
        spectral_centroid=(0.20, 0.60), spectral_rolloff=(0.25, 0.65),
        spectral_flatness=(0.0, 0.15), rms_var=(0.04, 0.22),
        onset_density=(0.55, 0.95), key_confidence=(0.40, 0.90),
    ),
    'house': dict(
        bpm=(120, 135), energy=(0.55, 0.92), beat_regularity=(0.75, 0.98),
        vocal_presence=(0.10, 0.55), breakdown_score=(0.30, 0.78),
        low_ratio=(0.28, 0.48), mid_ratio=(0.25, 0.42),
        spectral_centroid=(0.25, 0.65), spectral_rolloff=(0.28, 0.68),
        spectral_flatness=(0.02, 0.25), rms_var=(0.08, 0.32),
        onset_density=(0.35, 0.80), key_confidence=(0.55, 0.95),
    ),
    'pop': dict(
        bpm=(88, 132), energy=(0.38, 0.85), beat_regularity=(0.58, 0.90),
        vocal_presence=(0.55, 0.95), breakdown_score=(0.08, 0.48),
        low_ratio=(0.18, 0.38), mid_ratio=(0.30, 0.50),
        spectral_centroid=(0.35, 0.80), spectral_rolloff=(0.35, 0.80),
        spectral_flatness=(0.05, 0.35), rms_var=(0.18, 0.52),
        onset_density=(0.18, 0.58), key_confidence=(0.60, 0.98),
    ),
    'hiphop': dict(
        bpm=(68, 108), energy=(0.38, 0.80), beat_regularity=(0.48, 0.85),
        vocal_presence=(0.52, 0.92), breakdown_score=(0.08, 0.42),
        low_ratio=(0.35, 0.55), mid_ratio=(0.25, 0.42),
        spectral_centroid=(0.20, 0.60), spectral_rolloff=(0.22, 0.62),
        spectral_flatness=(0.03, 0.28), rms_var=(0.18, 0.55),
        onset_density=(0.18, 0.52), key_confidence=(0.45, 0.90),
    ),
    'rock': dict(
        bpm=(100, 160), energy=(0.50, 0.95), beat_regularity=(0.52, 0.85),
        vocal_presence=(0.40, 0.88), breakdown_score=(0.08, 0.48),
        low_ratio=(0.20, 0.40), mid_ratio=(0.30, 0.52),
        spectral_centroid=(0.30, 0.75), spectral_rolloff=(0.32, 0.78),
        spectral_flatness=(0.05, 0.38), rms_var=(0.20, 0.60),
        onset_density=(0.28, 0.72), key_confidence=(0.50, 0.92),
    ),
    'ambient': dict(
        bpm=(55, 95), energy=(0.08, 0.48), beat_regularity=(0.15, 0.62),
        vocal_presence=(0.0, 0.38), breakdown_score=(0.55, 0.98),
        low_ratio=(0.14, 0.32), mid_ratio=(0.30, 0.52),
        spectral_centroid=(0.15, 0.55), spectral_rolloff=(0.12, 0.52),
        spectral_flatness=(0.10, 0.55), rms_var=(0.28, 0.72),
        onset_density=(0.03, 0.28), key_confidence=(0.35, 0.80),
    ),
    'rnb': dict(
        bpm=(62, 100), energy=(0.32, 0.75), beat_regularity=(0.52, 0.85),
        vocal_presence=(0.62, 0.96), breakdown_score=(0.08, 0.42),
        low_ratio=(0.28, 0.48), mid_ratio=(0.28, 0.48),
        spectral_centroid=(0.25, 0.65), spectral_rolloff=(0.25, 0.65),
        spectral_flatness=(0.04, 0.30), rms_var=(0.18, 0.52),
        onset_density=(0.12, 0.42), key_confidence=(0.55, 0.95),
    ),
}

GENRE_NAMES = list(GENRES.keys())


def _rand(lo, hi):
    return np.random.uniform(lo, hi)


def generate_song(genre=None):
    if genre is None:
        genre = random.choice(GENRE_NAMES)

    p = GENRES[genre]

    bpm            = _rand(*p['bpm'])
    energy         = _rand(*p['energy'])
    beat_reg       = _rand(*p['beat_regularity'])
    vocal_presence = _rand(*p['vocal_presence'])
    breakdown      = _rand(*p['breakdown_score'])
    key_conf       = _rand(*p['key_confidence'])
    s_centroid     = _rand(*p['spectral_centroid'])
    s_rolloff      = _rand(*p['spectral_rolloff'])
    s_flatness     = _rand(*p['spectral_flatness'])
    rms_var        = _rand(*p['rms_var'])
    onset_density  = _rand(*p['onset_density'])

    low  = _rand(*p['low_ratio'])
    mid  = _rand(*p['mid_ratio'])
    high = max(0.05, 1.0 - low - mid)
    total = low + mid + high
    low, mid, high = low / total, mid / total, high / total

    camelot = random.choice(CAMELOT_KEYS)
    key, mode = CAMELOT_TO_INFO[camelot]
    key_pos = CIRCLE_OF_FIFTHS.index(key) / 12.0 if key in CIRCLE_OF_FIFTHS else 0.5

    features_array = np.array([
        bpm / 200.0,
        energy,
        key_pos,
        1.0 if mode == 'major' else 0.0,
        key_conf,
        low,
        mid,
        high,
        min(s_centroid, 1.0),
        min(s_rolloff,  1.0),
        min(s_flatness, 1.0),
        beat_reg,
        min(rms_var, 1.0),
        min(onset_density, 1.0),
    ], dtype=np.float32)

    return {
        'features_array': features_array,
        'bpm':             bpm,
        'energy':          energy,
        'key':             key,
        'mode':            mode,
        'camelot':         camelot,
        'key_confidence':  key_conf,
        'low_ratio':       low,
        'mid_ratio':       mid,
        'high_ratio':      high,
        'beat_regularity': beat_reg,
        'vocal_presence':  vocal_presence,
        'breakdown_score': breakdown,
    }


def generate_synthetic_dataset(n_pairs=10000, save_path='data/dataset/params_dataset.pkl'):
    print('=' * 60)
    print('  SYNTHETIC DATASET GENERATOR')
    print('=' * 60)
    print(f'  Genres:  {len(GENRES)} profiles')
    print(f'  Pairs:   {n_pairs}')
    print()

    ds = TransitionParamsDataset()

    genre_list = GENRE_NAMES

    for i in range(n_pairs):
        g1 = random.choice(genre_list)
        g2 = random.choice(genre_list)

        f1 = generate_song(g1)
        f2 = generate_song(g2)

        target = ds._compute_ideal_params(f1, f2)

        ds.data.append({
            'input':  np.concatenate([f1['features_array'], f2['features_array']]),
            'target': target,
        })

        if (i + 1) % 2000 == 0:
            print(f'  Generated {i + 1}/{n_pairs} pairs...')

    print(f'\n  Total samples: {len(ds.data)}')
    print(f'  Input shape:   {ds.data[0]["input"].shape}')
    print(f'  Target shape:  {ds.data[0]["target"].shape}')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ds.save(save_path)
    print(f'\n  Saved to: {save_path}')
    return ds


if __name__ == '__main__':
    generate_synthetic_dataset(n_pairs=10000)
    print('\n  Next step: python train_mel_encoder.py')
