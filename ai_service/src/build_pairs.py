import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(__file__))
FEAT_CSV = os.path.join(BASE, "data", "track_features.csv")
OUT_CSV = os.path.join(BASE, "data", "transition_pairs.csv")

df = pd.read_csv(FEAT_CSV)

KEY_ORDER = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
key_to_idx = {k:i for i,k in enumerate(KEY_ORDER)}

def key_distance(k1, k2):
    try:
        i1 = key_to_idx[k1]
        i2 = key_to_idx[k2]
        d = abs(i1 - i2)
        return min(d, 12 - d)
    except:
        return 6

pairs = []
filenames = df['filename'].tolist()
N = len(filenames)
print("Found", N, "tracks")

# ---- Create base pairs ----
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        a = df.iloc[i]
        b = df.iloc[j]

        delta_bpm = abs(a['bpm'] - b['bpm'])
        delta_key = key_distance(a['key'], b['key'])
        delta_energy = abs(a['energy'] - b['energy'])
        ratio_energy = (b['energy'] + 1e-9) / (a['energy'] + 1e-9)

        label = 1 if (delta_bpm <= 3 and delta_key <= 1 and delta_energy <= 0.1) else 0

        pairs.append({
            "file_A": a['filename'],
            "file_B": b['filename'],
            "delta_bpm": delta_bpm,
            "delta_key": delta_key,
            "delta_energy": delta_energy,
            "ratio_energy": ratio_energy,
            "label": label
        })

pairs_df = pd.DataFrame(pairs)

# ---- Check positives ----
min_positive = max(10, int(0.05 * len(pairs_df)))
pos_count = int(pairs_df['label'].sum())
print("Initial positives:", pos_count, "min desired:", min_positive)

# ---- Generate synthetic positives if needed ----
if pos_count < min_positive:
    synthetic_pairs = []
    print("Generating additional positive pairs...")

    for i in range(N):
        a = df.iloc[i]

        df_others = df.drop(i).copy()
        df_others['bpm_diff'] = (df_others['bpm'] - a['bpm']).abs()
        neighbors = df_others.sort_values('bpm_diff').head(2)

        for _, nb in neighbors.iterrows():
            synthetic_pairs.append({
                "file_A": a['filename'],
                "file_B": nb['filename'],
                "delta_bpm": abs(a['bpm'] - nb['bpm']),
                "delta_key": key_distance(a['key'], nb['key']),
                "delta_energy": abs(a['energy'] - nb['energy']),
                "ratio_energy": (nb['energy'] + 1e-9) / (a['energy'] + 1e-9),
                "label": 1
            })

    pairs_df = pd.concat([pairs_df, pd.DataFrame(synthetic_pairs)], ignore_index=True)

# ---- Save ----
pairs_df.to_csv(OUT_CSV, index=False)
print("Saved pairs to", OUT_CSV)
print("Label distribution:\n", pairs_df['label'].value_counts())
