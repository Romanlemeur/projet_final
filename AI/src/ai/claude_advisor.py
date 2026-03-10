"""
ClaudeTransitionAdvisor — Envoie l'analyse complète des deux morceaux à Claude
et reçoit en retour une décision structurée sur la transition :
  - Quand couper le morceau 1 (point sans voix)
  - Quand démarrer le morceau 2 (point sans voix)
  - Quel type de transition (smooth / drop / echo / quick)
  - Durée recommandée
  - Paramètres EQ suggérés
  - Raison de la décision
"""

import json
import numpy as np


class ClaudeTransitionAdvisor:
    """Utilise Claude pour décider intelligemment de la meilleure transition."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.hop_length = 512

    # ------------------------------------------------------------------
    # Helpers : conversion de la courbe vocale en timeline lisible
    # ------------------------------------------------------------------

    def build_vocal_timeline(self, vocal_curve: np.ndarray, sample_rate: int) -> list[dict]:
        """
        Convertit la courbe vocale frame-par-frame en liste seconde-par-seconde.
        Retourne : [{"second": 0, "vocal_score": 0.12, "has_vocals": false}, ...]
        """
        frames_per_sec = sample_rate / self.hop_length
        duration_sec = int(len(vocal_curve) / frames_per_sec)
        timeline = []

        for s in range(duration_sec):
            start = int(s * frames_per_sec)
            end = int((s + 1) * frames_per_sec)
            score = float(np.mean(vocal_curve[start:end])) if end > start else 0.0
            timeline.append({
                "second": s,
                "vocal_score": round(score, 3),
                "has_vocals": score > 0.32
            })

        return timeline

    def _find_instrumental_windows(self, timeline: list[dict], min_duration: int = 4) -> list[dict]:
        """Extrait les fenêtres continues sans voix depuis la timeline."""
        windows = []
        in_window = False
        start = 0

        for entry in timeline:
            if not entry["has_vocals"] and not in_window:
                start = entry["second"]
                in_window = True
            elif entry["has_vocals"] and in_window:
                duration = entry["second"] - start
                if duration >= min_duration:
                    windows.append({
                        "start": start,
                        "end": entry["second"],
                        "duration": duration,
                        "avg_vocal_score": round(
                            float(np.mean([e["vocal_score"] for e in timeline[start:entry["second"]]])), 3
                        )
                    })
                in_window = False

        if in_window:
            duration = len(timeline) - start
            if duration >= min_duration:
                windows.append({
                    "start": start,
                    "end": len(timeline),
                    "duration": duration,
                    "avg_vocal_score": round(
                        float(np.mean([e["vocal_score"] for e in timeline[start:]])), 3
                    )
                })

        return sorted(windows, key=lambda x: x["avg_vocal_score"])

    # ------------------------------------------------------------------
    # Appel Claude
    # ------------------------------------------------------------------

    def advise(
        self,
        analysis1: dict,
        analysis2: dict,
        vocal_curve1: np.ndarray,
        vocal_curve2: np.ndarray,
        sample_rate: int,
    ) -> dict:
        """
        Envoie toute l'analyse à Claude et retourne sa décision de transition.

        Retourne un dict avec :
          - cut_point_track1 : float (secondes)
          - start_point_track2 : float (secondes)
          - style : str (smooth / drop / echo / quick)
          - transition_duration : float (secondes)
          - eq_adjustments : dict
          - reasoning : str
          - confidence : float (0-1)
        """
        print("\n  [Claude] Construction de la timeline vocale...")
        timeline1 = self.build_vocal_timeline(vocal_curve1, sample_rate)
        timeline2 = self.build_vocal_timeline(vocal_curve2, sample_rate)

        windows1 = self._find_instrumental_windows(timeline1, min_duration=4)
        windows2 = self._find_instrumental_windows(timeline2, min_duration=4)

        print(f"  [Claude] Track 1 : {len(windows1)} fenêtres instrumentales")
        print(f"  [Claude] Track 2 : {len(windows2)} fenêtres instrumentales")

        payload = self._build_payload(
            analysis1, analysis2,
            timeline1, timeline2,
            windows1, windows2
        )

        print("  [Claude] Envoi de l'analyse...")
        prompt = self._build_prompt(payload)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()
        print(f"  [Claude] Réponse reçue ({len(raw)} chars)")

        return self._parse_response(raw, analysis1, analysis2, windows1, windows2)

    def _build_payload(self, a1, a2, tl1, tl2, w1, w2) -> dict:
        """Construit le payload JSON envoyé à Claude."""
        return {
            "track1": {
                "duration": round(a1.get("duration", 0), 1),
                "bpm": round(a1.get("tempo", 120), 1),
                "key": f"{a1.get('key', 'C')} {a1.get('mode', 'major')}",
                "camelot": a1.get("camelot", "8B"),
                "sections": [
                    {"type": s["type"], "start": round(s["start"], 1), "end": round(s["end"], 1)}
                    for s in a1.get("sections", [])
                ],
                "vocal_timeline": tl1,
                "instrumental_windows": w1[:8],
                "vocal_segments_count": len(a1.get("vocal_segments", [])),
            },
            "track2": {
                "duration": round(a2.get("duration", 0), 1),
                "bpm": round(a2.get("tempo", 120), 1),
                "key": f"{a2.get('key', 'C')} {a2.get('mode', 'major')}",
                "camelot": a2.get("camelot", "8A"),
                "sections": [
                    {"type": s["type"], "start": round(s["start"], 1), "end": round(s["end"], 1)}
                    for s in a2.get("sections", [])
                ],
                "vocal_timeline": tl2,
                "instrumental_windows": w2[:8],
                "vocal_segments_count": len(a2.get("vocal_segments", [])),
            }
        }

    def _build_prompt(self, payload: dict) -> str:
        data_json = json.dumps(payload, ensure_ascii=False)

        return f"""Tu es un DJ expert en mixage professionnel. Analyse ces deux morceaux et décide comment faire la meilleure transition possible.

DONNÉES D'ANALYSE :
{data_json}

RÈGLES IMPORTANTES :
1. Ne jamais couper le morceau 1 quand "has_vocals" est true (la voix doit être terminée)
2. Ne jamais commencer le morceau 2 quand "has_vocals" est true (la voix ne doit pas encore avoir commencé)
3. Préférer les fenêtres instrumentales (instrumental_windows) avec le score vocal le plus bas
4. Le cut_point_track1 doit être dans le dernier tiers de la piste 1 (après 60% de la durée)
5. Le start_point_track2 doit être dans le premier tiers de la piste 2 (avant 30% de la durée)
6. Tenir compte de la compatibilité harmonique (Camelot wheel) pour choisir le style
7. Tenir compte de la différence de BPM pour la durée de transition

STYLES DISPONIBLES :
- "smooth" : crossfade doux et atmosphérique (idéal si compatibilité harmonique > 80%)
- "drop" : build-up énergique avec drop (idéal pour transitions rapides ou montée d'énergie)
- "echo" : queue de réverbe éthérée (idéal pour créer de la distance entre deux morceaux)
- "quick" : transition rapide sur une mesure (idéal si BPM très différents ou clash harmonique)

Réponds UNIQUEMENT avec un JSON valide, sans texte avant ou après :
{{
  "cut_point_track1": <float, secondes depuis le début du morceau 1>,
  "start_point_track2": <float, secondes depuis le début du morceau 2>,
  "style": "<smooth|drop|echo|quick>",
  "transition_duration": <float, durée en secondes entre 8 et 48>,
  "eq_adjustments": {{
    "bass_swap_position": <float 0.0-1.0, position du swap de basses dans la transition>,
    "filter_sweep": <float 0.0-1.0, intensité du filter sweep>,
    "low_cut_track1": <float en dB, -12 à 0>,
    "low_cut_track2": <float en dB, -12 à 0>
  }},
  "reasoning": "<explication courte de la décision en français>",
  "confidence": <float 0.0-1.0>
}}"""

    def _parse_response(self, raw: str, a1: dict, a2: dict, w1: list, w2: list) -> dict:
        """Parse la réponse JSON de Claude, avec fallback si invalide."""
        try:
            # Extraire le JSON si Claude a ajouté du texte autour
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]

            decision = json.loads(raw)

            # Valider et corriger les valeurs
            duration1 = a1.get("duration", 180)
            duration2 = a2.get("duration", 180)

            cut = float(decision.get("cut_point_track1", duration1 * 0.75))
            cut = max(duration1 * 0.5, min(cut, duration1 - 10))

            start_pt = float(decision.get("start_point_track2", 10.0))
            start_pt = max(0, min(start_pt, duration2 * 0.3))

            trans_dur = float(decision.get("transition_duration", 20.0))
            trans_dur = max(8.0, min(trans_dur, 48.0))

            style = decision.get("style", "smooth")
            if style not in ("smooth", "drop", "echo", "quick"):
                style = "smooth"

            eq = decision.get("eq_adjustments", {})

            print(f"  [Claude] Cut track 1 : {cut:.1f}s")
            print(f"  [Claude] Start track 2 : {start_pt:.1f}s")
            print(f"  [Claude] Style : {style} | Durée : {trans_dur:.0f}s")
            print(f"  [Claude] Raison : {decision.get('reasoning', '')}")

            return {
                "cut_point_track1": cut,
                "start_point_track2": start_pt,
                "style": style,
                "transition_duration": trans_dur,
                "eq_adjustments": {
                    "bass_swap_position": float(eq.get("bass_swap_position", 0.5)),
                    "filter_sweep": float(eq.get("filter_sweep", 0.4)),
                    "low_cut_track1": float(eq.get("low_cut_track1", -6.0)),
                    "low_cut_track2": float(eq.get("low_cut_track2", -6.0)),
                },
                "reasoning": decision.get("reasoning", ""),
                "confidence": float(decision.get("confidence", 0.7)),
                "claude_used": True,
            }

        except Exception as e:
            print(f"  [Claude] Erreur parsing ({e}) — fallback sur valeurs par défaut")
            return self._fallback_decision(a1, a2, w1, w2)

    def _fallback_decision(self, a1: dict, a2: dict, w1: list, w2: list) -> dict:
        """Décision de secours si Claude échoue."""
        duration1 = a1.get("duration", 180)
        duration2 = a2.get("duration", 180)

        # Meilleure fenêtre instrumentale dans le dernier tiers du morceau 1
        outro_windows = [w for w in w1 if w["start"] > duration1 * 0.55]
        cut = outro_windows[0]["start"] + outro_windows[0]["duration"] * 0.5 if outro_windows else duration1 * 0.75

        # Meilleure fenêtre dans le premier tiers du morceau 2
        intro_windows = [w for w in w2 if w["end"] < duration2 * 0.4]
        start_pt = intro_windows[0]["start"] if intro_windows else 8.0

        return {
            "cut_point_track1": cut,
            "start_point_track2": start_pt,
            "style": "smooth",
            "transition_duration": 20.0,
            "eq_adjustments": {
                "bass_swap_position": 0.5,
                "filter_sweep": 0.4,
                "low_cut_track1": -6.0,
                "low_cut_track2": -6.0,
            },
            "reasoning": "Fallback : fenêtres instrumentales détectées automatiquement",
            "confidence": 0.5,
            "claude_used": False,
        }
