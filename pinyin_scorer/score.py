"""Score one recording of a single syllable against the syllable the learner was attempting.

For each part (initial, final, tone) the model answers: assuming exactly one unit was
spoken, how likely is each candidate? That gives a score for the target and, when the
target loses, what was heard instead.

Command line:
    python score.py --model runs/stage2/best --audio attempt.wav --target zhuang4

In a server, load once and reuse:
    scorer = Scorer("runs/stage2/best")
    result = scorer.score("attempt.wav", "zhuang4")
"""
import argparse
import json

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


class Scorer:
    def __init__(self, model_dir, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = PinyinCTC.load(model_dir).to(self.device).eval()
        self.fe = PinyinCTC.load_feature_extractor(model_dir)

    @torch.no_grad()
    def score(self, audio, target, threshold=DEFAULT_THRESHOLD, top_k=3):
        """audio: a file path, or a mono float32 numpy array at 16 kHz.

        A part counts as correct unless the model gives the target less than `threshold`
        probability. "heard" is still the model's top guess, so it can differ from the
        target on a part that is marked correct; that means "close enough"."""
        parts = split_syllable(target)
        if parts is None:
            raise ValueError(f"not a pinyin syllable with tone: {target!r}")
        wav = load_audio(audio) if isinstance(audio, str) else pad_audio(
            normalize_loudness(np.asarray(audio, dtype=np.float32)))
        feats = self.fe(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        logits = self.model(feats["input_values"].to(self.device))

        result = {"target": target, "parts": {}}
        silence_probs = []
        for head, want in zip(HEADS, parts):
            log_probs = F.log_softmax(logits[head][0].float(), dim=-1).cpu()
            scores, silence = single_unit_scores(log_probs)
            probs = torch.softmax(torch.cat([scores, silence.view(1)]), dim=0)
            silence_probs.append(probs[-1].item())
            units = VOCABS[head][1:]
            ranked = sorted(zip(units, probs[:-1].tolist()), key=lambda x: -x[1])
            score = dict(ranked)[want]
            heard = ranked[0][0]
            result["parts"][head] = {
                "target": display(head, want),
                "heard": display(head, heard),
                "target_unit": want,
                "heard_unit": heard,
                "score": round(score, 4),
                "correct": score >= threshold,
                "top": [(display(head, u), round(pr, 4)) for u, pr in ranked[:top_k]],
            }
        result["no_speech"] = min(silence_probs) > 0.5
        result["understood"] = (not result["no_speech"]
                                and all(p["correct"] for p in result["parts"].values()))
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--target", required=True, help="e.g. zhuang4, lv4, yi1")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = p.parse_args()
    result = Scorer(args.model).score(args.audio, args.target, args.threshold)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()