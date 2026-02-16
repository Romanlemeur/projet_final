import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "transition_model_rf.joblib"

model = joblib.load(MODEL_PATH)

def score_pair_from_deltas(delta_bpm, delta_key, delta_energy, ratio_energy):
    X = [[delta_bpm, delta_key, delta_energy, ratio_energy]]
    prob = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]
    return {"score": float(prob), "prediction": int(pred)}

def score_pair(trackA, trackB):
    # trackA / trackB doivent contenir: bpm, energy, key
    delta_bpm = abs(trackA["bpm"] - trackB["bpm"])

    # Key distance simple : si même key -> 0 sinon 6 (approx) pour rester cohérent
    delta_key = 0 if trackA["key"] == trackB["key"] else 6

    delta_energy = abs(trackA["energy"] - trackB["energy"])
    ratio_energy = (trackB["energy"] + 1e-9) / (trackA["energy"] + 1e-9)

    return score_pair_from_deltas(delta_bpm, delta_key, delta_energy, ratio_energy)
