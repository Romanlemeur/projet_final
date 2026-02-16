

from src.inference import score_pair

trackA = {
    "bpm": 120,
    "energy": 0.65,
    "spectral_centroid": 2100,
    "key": "A"
}

trackB = {
    "bpm": 124,
    "energy": 0.68,
    "spectral_centroid": 2200,
    "key": "A"
}

print(score_pair(trackA, trackB))

