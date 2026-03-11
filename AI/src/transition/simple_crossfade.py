import numpy as np


def simple_crossfade(audio1, audio2, sample_rate, transition_duration=15.0):
    n_trans = int(transition_duration * sample_rate)
    n_trans = min(n_trans, len(audio1), len(audio2))

    t = np.linspace(0, 1, n_trans)
    crossfade = audio1[-n_trans:] * (1 - t) + audio2[:n_trans] * t

    parts = []
    if len(audio1) > n_trans:
        parts.append(audio1[:-n_trans])
    parts.append(crossfade)
    if len(audio2) > n_trans:
        parts.append(audio2[n_trans:])

    full = np.concatenate(parts)

    peak = np.max(np.abs(full))
    if peak > 0:
        full = full * (0.95 / peak)

    return full
