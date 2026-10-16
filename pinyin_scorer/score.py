"""Score one recording of a single syllable against the syllable the learner was attempting.

For each part (initial, final, tone) the model answers: assuming exactly one unit was
spoken, how likely is each candidate? That gives a score for the target and, when the
target loses, what was heard instead.

Command line:
    python score.py --model runs/stage6b/best runs/stage7_hubert/best \
        --pitch-model runs/tone23_pitch.joblib --audio attempt.wav --target zhuang4

In a server, load once and reuse:
    scorer = Scorer(["runs/stage6b/best", "runs/stage7_hubert/best"],
                    pitch_model="runs/tone23_pitch.joblib")
    result = scorer.score("attempt.wav", "zhuang4")

Several model folders are combined by averaging their probabilities per part. With a pitch model,
the pitch contour re-splits the tone 2 vs tone 3 probability (see PITCH_WEIGHT).
"""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from data import SAMPLE_RATE, load_audio, normalize_loudness, pad_audio
from model import PinyinCTC
from pinyin_units import HEADS, VOCABS, display, split_syllable


# A part is flagged as wrong only when the target's probability falls below this.
# Measured on native speech (stage 6b, voices it never trained on): at 0.5 about 18% of
# correct syllables were wrongly flagged, at 0.05 about 7.5%, while swapped-target errors
# on easily confused pairs slipped through 1 to 3% of the time.
DEFAULT_THRESHOLD = 0.05

# Answers accepted in place of the target, per head. The partner's probability is added to the
# target's before the threshold is applied, so a clip heard as the partner still counts as correct.
# One-directional unless listed both ways: eng accepts ang, but ang does not accept eng.
# eng/ang: the model can't reliably tell an open eng from ang on unfamiliar voices (Oct 2026), and
# borderline eng is often accepted by native listeners when the intended word is known.
# n/ng swaps (product decision): an<->ang, en<->eng, in<->ing, ian<->iang, uan<->uang.
# Vowel swaps (one-directional): eng heard as ang, en heard as an.
FORGIVE = {
    "final": {
        "an": ("ang",), "ang": ("an",),
        "en": ("eng", "an"), "eng": ("en", "ang"),
        "in": ("ing",), "ing": ("in",),
        "ian": ("iang",), "iang": ("ian",),
        "uan": ("uang",), "uang": ("uan",),
    },
}

# How much the pitch contour counts when re-splitting tone 2 vs 3 (0 = model only, 1 = pitch only).
# 0.3 measured Oct 2026 (6b+7, web voices): false alarms 3.3% -> 1.9%, all-3 accuracy 90.3% -> 91.6%,
# while native tone 3 accepted as an attempt at tone 2 rose 6.9% -> 8.8%. Above 0.3 that cost rises fast.
PITCH_WEIGHT = 0.3


def single_unit_scores(log_probs):
    """log_probs: (T, C) with blank at 0.

    Returns (scores, silence): the log-probability of every one-unit CTC path
    (blank* unit+ blank*) for each unit, and of the all-blank path.
    """
    blank, unit = log_probs[:, 0], log_probs[:, 1:]
    before = blank[0]                                   # still in leading blanks
    inside = unit[0].clone()                            # currently emitting the unit
    after = torch.full_like(inside, float("-inf"))      # in trailing blanks
    for t in range(1, log_probs.shape[0]):
        new_inside = torch.logaddexp(before, inside) + unit[t]
        new_after = torch.logaddexp(inside, after) + blank[t]
        before = before + blank[t]
        inside, after = new_inside, new_after
    return torch.logaddexp(inside, after), before


# ---------------------------------------------------------------- pure functions (no torch), also used
# by the evaluation notebook so it measures exactly what the app does.
# A "probs" object is {head: {unit: probability, ..., "<silence>": probability}}.

def combine(prob_list):
    """Average several models' probs."""
    return {h: {u: sum(p[h][u] for p in prob_list) / len(prob_list) for u in prob_list[0][h]}
            for h in prob_list[0]}


def apply_pitch(probs, vote, weight=PITCH_WEIGHT):
    """Re-split the tone 2 + tone 3 probability using the pitch vote (log-odds of 2 over 3).
    Their total, and every other tone, stays as the model gave it."""
    if vote is None or weight <= 0:
        return probs
    tone = dict(probs["tone"])
    m2, m3 = tone["2"], tone["3"]
    lm = math.log(m2 + 1e-9) - math.log(m3 + 1e-9)
    z = (1 - weight) * lm + weight * vote
    q = 1 / (1 + math.exp(-max(-50.0, min(50.0, z))))
    tone["2"], tone["3"] = (m2 + m3) * q, (m2 + m3) * (1 - q)
    return {**probs, "tone": tone}


def credited(probs, head, unit, forgive=FORGIVE):
    """Probability counted for `unit`: its own plus that of any forgiven partner."""
    d = probs[head]
    return d[unit] + sum(d[u] for u in forgive.get(head, {}).get(unit, ()))


def judge(probs, target, threshold=DEFAULT_THRESHOLD, top_k=3, forgive=FORGIVE):
    """The result dict the app receives, from (combined) probs."""
    parts = split_syllable(target)
    if parts is None:
        raise ValueError(f"not a pinyin syllable with tone: {target!r}")
    result = {"target": target, "parts": {}}
    silence_probs = []
    for head, want in zip(HEADS, parts):
        d = probs[head]
        silence_probs.append(d["<silence>"])
        ranked = sorted(((u, d[u]) for u in VOCABS[head][1:]), key=lambda x: -x[1])
        forgiven = forgive.get(head, {}).get(want, ())
        score = credited(probs, head, want, forgive)
        heard = ranked[0][0]
        result["parts"][head] = {
            "target": display(head, want),
            "heard": display(head, heard),
            "target_unit": want,
            "heard_unit": heard,
            "score": round(score, 4),
            "correct": score >= threshold,
            "forgiven": heard in forgiven,   # top guess was an accepted partner, e.g. ang for eng
            "top": [(display(head, u), round(pr, 4)) for u, pr in ranked[:top_k]],
        }
    result["no_speech"] = min(silence_probs) > 0.5
    result["understood"] = (not result["no_speech"]
                            and all(p["correct"] for p in result["parts"].values()))
    return result


# ---------------------------------------------------------------- the scorer
class Scorer:
    def __init__(self, model_dirs, device=None, pitch_model=None, pitch_weight=PITCH_WEIGHT,
                 quantize=False, threads=None):
        """model_dirs: one model folder or a list of them (probabilities are averaged).
        pitch_model: path to the saved tone 2/3 pitch classifier, or None to skip it.
        quantize: store the models' linear layers as 8-bit integers (CPU only). About 4x less memory
            for the weights and usually faster on CPU; check accuracy with quant_check.py first.
        threads: CPU threads for PyTorch (e.g. the number of cores on the server)."""
        import gc
        if isinstance(model_dirs, str):
            model_dirs = [model_dirs]
        if threads:
            torch.set_num_threads(threads)
        if quantize:
            device = "cpu"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models = []
        for m in model_dirs:                      # one at a time, so peak memory stays low
            net = PinyinCTC.load(m).eval()
            if quantize:
                net = torch.quantization.quantize_dynamic(net, {torch.nn.Linear}, dtype=torch.qint8)
            self.models.append((net.to(self.device), PinyinCTC.load_feature_extractor(m)))
            gc.collect()
        self.model, self.fe = self.models[0]      # for older scripts that use these directly
        self.pitch = None
        if pitch_model:
            from pitch_tone import PitchTone
            self.pitch = PitchTone(pitch_model)
        self.pitch_weight = pitch_weight

    @staticmethod
    def prepare(audio):
        """A file path, or a mono float32 numpy array at 16 kHz -> the waveform the models see."""
        return load_audio(audio) if isinstance(audio, str) else pad_audio(
            normalize_loudness(np.asarray(audio, dtype=np.float32)))

    @torch.no_grad()
    def model_probs(self, wav):
        """One probs object per model (before combining or pitch)."""
        out = []
        for model, fe in self.models:
            feats = fe(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            logits = model(feats["input_values"].to(self.device))
            per_head = {}
            for head in HEADS:
                log_probs = F.log_softmax(logits[head][0].float(), dim=-1).cpu()
                scores, silence = single_unit_scores(log_probs)
                p = torch.softmax(torch.cat([scores, silence.view(1)]), dim=0).tolist()
                per_head[head] = dict(zip(VOCABS[head][1:], p[:-1]))
                per_head[head]["<silence>"] = p[-1]
            out.append(per_head)
        return out

    def pitch_vote(self, wav):
        return self.pitch.vote(wav) if self.pitch is not None else None

    def probs(self, audio):
        wav = self.prepare(audio)
        return apply_pitch(combine(self.model_probs(wav)), self.pitch_vote(wav), self.pitch_weight)

    def score(self, audio, target, threshold=DEFAULT_THRESHOLD, top_k=3):
        """A part counts as correct unless the target (plus any forgiven partner) gets less than
        `threshold` probability. "heard" is still the top guess, so it can differ from the target on
        a part that is marked correct; that means "close enough" (or "forgiven")."""
        return judge(self.probs(audio), target, threshold, top_k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs="+", required=True, help="one or more model folders")
    p.add_argument("--pitch-model", default=None, help="saved tone 2/3 pitch classifier (.joblib)")
    p.add_argument("--audio", required=True)
    p.add_argument("--target", required=True, help="e.g. zhuang4, lv4, yi1")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--quantize", action="store_true", help="8-bit linear layers, CPU only")
    args = p.parse_args()
    result = Scorer(args.model, pitch_model=args.pitch_model, quantize=args.quantize).score(
        args.audio, args.target, args.threshold)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()