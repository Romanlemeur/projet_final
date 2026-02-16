import numpy as np
import librosa

#  BPM 
def extract_bpm(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)

#  Energy 
def extract_energy(y):
    rms = librosa.feature.rms(y=y)[0]
    return float(np.mean(rms))

#  Spectral 
def extract_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    return float(np.mean(centroid))

#  KEY detection 
def estimate_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                 'F#', 'G', 'G#', 'A', 'A#', 'B']

    key_index = np.argmax(chroma_mean)
    return key_names[key_index]

#  ALL FEATURES
def extract_features(y, sr):
    return {
        "bpm": extract_bpm(y, sr),
        "energy": extract_energy(y),
        "spectral_centroid": extract_spectral_centroid(y, sr),
        "key": estimate_key(y, sr)
    }
