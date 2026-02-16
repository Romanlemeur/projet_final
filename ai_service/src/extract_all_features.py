
import os
import json
import pandas as pd
from audio_loader import load_audio
from feature_extractor import extract_features

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUT_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "track_features.csv")

rows = []
for fname in os.listdir(DATA_DIR):
    if not fname.lower().endswith(".mp3"):
        continue
    path = os.path.join(DATA_DIR, fname)
    print("Processing", fname)
    try:
        y, sr = load_audio(path)
        feats = extract_features(y, sr)
        feats["filename"] = fname
        rows.append(feats)
    except Exception as e:
        print("ERROR", fname, e)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print("Saved features to", OUT_CSV)
print(df.head())
