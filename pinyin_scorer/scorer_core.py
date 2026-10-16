"""The scoring rules and audio preparation, with no PyTorch.

Shared by score.py (PyTorch models, used in Colab) and onnx_scorer.py (ONNX models, used on the
server), so both make exactly the same decisions. The audio functions are copies of data.py's, so
the server sees exactly what the models were trained on without importing torch.

A "probs" object is {head: {unit: probability, ..., "<silence>": probability}}.
"""
import math
from math import gcd

import numpy as np

from pinyin_units import HEADS, VOCABS, display, split_syllable

SAMPLE_RATE = 16000
TARGET_DBFS = -20.0
MAX_GAIN_DB = 40.0

# A part is flagged as wrong only when the target's probability falls below this.
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


# ---------------------------------------------------------------- audio (same as data.py)
def speech_power(wav, frame=160):
    n = len(wav) // frame
    if n == 0:
        return float(np.mean(wav ** 2)) + 1e-10
    power = (wav[:n * frame].reshape(n, frame) ** 2).mean(axis=1)
    loud = np.sort(power)[-max(1, int(0.3 * n)):]
    return float(loud.mean()) + 1e-10


def normalize_loudness(wav, target_dbfs=TARGET_DBFS):
    gain_db = target_dbfs - 10 * np.log10(speech_power(wav))
    wav = wav * 10 ** (min(gain_db, MAX_GAIN_DB) / 20)
    peak = float(np.abs(wav).max()) if len(wav) else 0.0
    if peak > 0.99:
        wav = wav * (0.99 / peak)
    return wav.astype(np.float32)


def pad_audio(wav, min_seconds=0.5):
    short = int(min_seconds * SAMPLE_RATE) - len(wav)
    if short > 0:
        wav = np.pad(wav, (short // 2, short - short // 2))
    return wav


def load_audio(path, min_seconds=0.5):
    import soundfile as sf
    from scipy.signal import resample_poly
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = gcd(sr, SAMPLE_RATE)
        wav = resample_poly(wav, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return pad_audio(normalize_loudness(wav), min_seconds)


def prepare(audio):
    """A file path, or a mono float32 numpy array at 16 kHz -> the waveform the models see."""
    return load_audio(audio) if isinstance(audio, str) else pad_audio(
        normalize_loudness(np.asarray(audio, dtype=np.float32)))


# ---------------------------------------------------------------- model output -> probs
def single_unit_scores_np(log_probs):
    """numpy version of score.single_unit_scores. log_probs: (T, C), blank at 0.
    Returns (log-prob of every one-unit CTC path per unit, log-prob of the all-blank path)."""
    blank, unit = log_probs[:, 0], log_probs[:, 1:]
    before = blank[0]
    inside = unit[0].copy()
    after = np.full_like(inside, -np.inf)
    for t in range(1, log_probs.shape[0]):
        new_inside = np.logaddexp(before, inside) + unit[t]
        new_after = np.logaddexp(inside, after) + blank[t]
        before = before + blank[t]
        inside, after = new_inside, new_after
    return np.logaddexp(inside, after), before


def head_probs(logits, head):
    """logits: (T, C) for one head -> {unit: probability, "<silence>": probability}."""
    x = logits.astype(np.float64)
    lp = x - np.logaddexp.reduce(x, axis=-1, keepdims=True)
    scores, silence = single_unit_scores_np(lp)
    z = np.concatenate([scores, [silence]])
    p = np.exp(z - z.max())
    p /= p.sum()
    out = dict(zip(VOCABS[head][1:], p[:-1].tolist()))
    out["<silence>"] = float(p[-1])
    return out


# ---------------------------------------------------------------- decisions
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