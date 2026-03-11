CROSSFADE_TYPES = {0: "linear", 1: "cosine (S-curve)", 2: "exponential"}
MIX_STYLES = {0: "smooth blend", 1: "classic DJ mix", 2: "cut mix", 3: "build-up/drop", 4: "tension build"}
TENSION_NAMES = {0: "none", 1: "echo", 2: "reverb wash", 3: "noise sweep"}


def _fmt_time(seconds):
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def explain_params(params, analysis1, analysis2, harmony, cut_info):
    decisions = []

    cam1 = analysis1.get('camelot', '?')
    cam2 = analysis2.get('camelot', '?')
    harm_score = harmony.get('score', 0)
    harm_type = harmony.get('type', 'unknown')
    decisions.append(
        f"Key compatibility: {cam1} -> {cam2} ({harm_type}, {harm_score:.0%} match)"
    )

    bpm1 = analysis1.get('tempo', 0)
    bpm2 = analysis2.get('tempo', 0)
    bpm_diff_pct = abs(bpm1 - bpm2) / max(bpm1, bpm2) * 100 if max(bpm1, bpm2) > 0 else 0
    if bpm_diff_pct > 2.5:
        decisions.append(
            f"BPM mismatch: {bpm1:.1f} -> {bpm2:.1f} ({bpm_diff_pct:.1f}% diff) - time-stretch applied"
        )
    else:
        decisions.append(
            f"BPM compatible: {bpm1:.1f} -> {bpm2:.1f} - no time-stretch needed"
        )

    duck1 = params.get('duck_vocals_1', 0)
    duck2 = params.get('duck_vocals_2', 0)
    if duck1 > 0.5 and duck2 > 0.4:
        decisions.append(
            f"Vocals detected in both tracks - heavy ducking applied "
            f"(Track A: {duck1:.0%}, Track B: {duck2:.0%}) to prevent clash"
        )
    elif duck1 > 0.2 or duck2 > 0.2:
        decisions.append(
            f"Partial vocal presence - mild ducking applied "
            f"(Track A: {duck1:.0%}, Track B: {duck2:.0%})"
        )
    else:
        decisions.append("No significant vocals detected - full mix without ducking")

    loop1 = cut_info.get('loop1', {})
    loop2 = cut_info.get('loop2', {})
    loop1_start = loop1.get('start', 0)
    loop2_start = loop2.get('start', 0)
    decisions.append(
        f"Outro loop selected at {_fmt_time(loop1_start)} in Track A"
        + (" (breakdown zone)" if not loop1.get('has_vocals', True) else " (vocals present - managed)")
    )
    decisions.append(
        f"Intro loop selected at {_fmt_time(loop2_start)} in Track B"
        + (" (clean intro)" if not loop2.get('has_vocals', True) else " (vocals present - managed)")
    )

    cf_type = int(params.get('crossfade_type', 1))
    cf_name = CROSSFADE_TYPES.get(cf_type, 'cosine')
    decisions.append(f"Crossfade curve: {cf_name}")

    mix_style = int(params.get('mix_style', 0))
    style_name = MIX_STYLES.get(mix_style, 'smooth blend')
    decisions.append(f"Mix style: {style_name}")

    t_beats = int(params.get('transition_beats', 32))
    avg_bpm = (bpm1 + bpm2) / 2 if bpm1 and bpm2 else 128
    t_secs = (t_beats / avg_bpm) * 60
    decisions.append(f"Transition length: {t_beats} beats ({t_secs:.1f}s at {avg_bpm:.0f} avg BPM)")

    bass_swap = params.get('bass_swap_beat', 0.5)
    decisions.append(
        f"Bass swap at {bass_swap:.0%} through transition "
        f"({_fmt_time(t_secs * bass_swap)} mark)"
    )

    low_eq1 = params.get('low_eq_1', 0)
    low_eq2 = params.get('low_eq_2', 0)
    decisions.append(
        f"Low EQ: Track A {low_eq1:+.2f}dB, Track B {low_eq2:+.2f}dB "
        f"(avoids bass buildup during overlap)"
    )

    filter_sweep = params.get('filter_sweep', 0)
    if filter_sweep > 0.6:
        decisions.append("Filter sweep: high - builds tension leading into drop")
    elif filter_sweep > 0.3:
        decisions.append("Filter sweep: moderate - adds energy movement")
    else:
        decisions.append("Filter sweep: minimal - clean transparent mix")

    tension = int(params.get('tension_effect', 0))
    tension_name = TENSION_NAMES.get(tension, 'none')
    if tension > 0:
        decisions.append(f"Tension effect: {tension_name} applied during transition")

    energy_dir = params.get('energy_direction', 0.5)
    e1 = analysis1.get('tempo', 128)
    if energy_dir > 0.6:
        decisions.append("Energy direction: upward - mix drives into higher energy")
    elif energy_dir < 0.4:
        decisions.append("Energy direction: downward - mix winds down smoothly")
    else:
        decisions.append("Energy direction: neutral - maintained throughout")

    return decisions
