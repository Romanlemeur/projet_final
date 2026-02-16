import json
import os
from audio_loader import load_audio
from feature_extractor import extract_features

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base, "data", "sample_track.mp3")

    y, sr = load_audio(path)
    features = extract_features(y, sr)

    print("Features:")
    print(json.dumps(features, indent=2))
