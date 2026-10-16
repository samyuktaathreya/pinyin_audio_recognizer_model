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

import numpy as np
import torch
import torch.nn.functional as F

from data import SAMPLE_RATE, load_audio, normalize_loudness, pad_audio
from model import PinyinCTC
from pinyin_units import HEADS, VOCABS, display, split_syllable


# Rules, thresholds and the decision functions live in scorer_core.py (no PyTorch), shared with
# the ONNX server. Re-exported here so `from score import FORGIVE, judge, ...` keeps working.
from scorer_core import (DEFAULT_THRESHOLD, FORGIVE, PITCH_WEIGHT, apply_pitch, combine,  # noqa: F401
                         credited, judge)


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