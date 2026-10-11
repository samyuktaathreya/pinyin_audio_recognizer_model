"""Save the model's full scores for every clip to one CSV, for calibration analysis.

eval_sources.py only keeps the top guess. This keeps everything: for each clip, and for
each part (initial, final, tone), the log-probability of every candidate unit and of
silence, computed exactly the way score.py computes them. With that file, any decision
rule (thresholds, per-pair leniency, swapped-target tests) can be tried afterwards
without running the model again.

    python dump_probs.py --model /content/drive/MyDrive/pinyin_runs/stage2/best \
        --manifests manifests/ext/hotoo.jsonl manifests/ext/studycli.jsonl ... \
        --out /content/drive/MyDrive/pinyin_runs/stage2_probs.csv

Put this file next to score.py. It does not change any existing file.
Rows are written as they are scored, so a disconnect loses nothing already done.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data import SAMPLE_RATE, load_audio
from pinyin_units import HEADS, VOCABS, split_syllable
from score import Scorer, single_unit_scores

SILENCE = "<silence>"


def unit_names(head):
    """Candidate names for one head, in model order, with silence last."""
    return [str(u) for u in VOCABS[head][1:]] + [SILENCE]


def all_columns():
    cols = ["source", "audio", "target", "seconds"] + [f"target_{h}" for h in HEADS]
    for h in HEADS:
        cols += [f"{h}:{name}" for name in unit_names(h)]
    return cols


@torch.no_grad()
def clip_scores(scorer, path):
    """Returns ({column: log-probability}, clip length in seconds)."""
    wav = load_audio(path)
    feats = scorer.fe(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    logits = scorer.model(feats["input_values"].to(scorer.device))
    out = {}
    for h in HEADS:
        log_probs = F.log_softmax(logits[h][0].float(), dim=-1).cpu()
        scores, silence = single_unit_scores(log_probs)
        # Same normalisation as score.py, kept in log form so that very confident
        # answers are not rounded to exactly 0 or 1.
        logp = torch.log_softmax(torch.cat([scores, silence.view(1)]), dim=0).tolist()
        for name, value in zip(unit_names(h), logp):
            out[f"{h}:{name}"] = round(value, 4)
    return out, len(wav) / SAMPLE_RATE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--manifests", nargs="+", required=True)
    p.add_argument("--limit", type=int, default=0, help="max clips per manifest; 0 = all")
    p.add_argument("--out", default="probs.csv")
    args = p.parse_args()

    scorer = Scorer(args.model)
    written = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns())
        writer.writeheader()
        for manifest in args.manifests:
            source = Path(manifest).stem
            rows = [json.loads(line) for line in open(manifest, encoding="utf-8") if line.strip()]
            rows = [r for r in rows if split_syllable(r["pinyin"])]
            if args.limit:
                rows = rows[:args.limit]
            done = 0
            for r in rows:
                try:
                    scores, seconds = clip_scores(scorer, r["audio"])
                except Exception as e:
                    print(f"   could not score {r['audio']}: {e}")
                    continue
                parts = split_syllable(r["pinyin"])
                writer.writerow({
                    "source": source, "audio": r["audio"], "target": r["pinyin"],
                    "seconds": round(seconds, 3),
                    **{f"target_{h}": str(want) for h, want in zip(HEADS, parts)},
                    **scores,
                })
                done += 1
                if done % 200 == 0:
                    f.flush()
                    print(f"   {source}: {done}/{len(rows)}", flush=True)
            f.flush()
            written += done
            print(f"{source:16} {done:6d} clips", flush=True)
    print(f"\n{written} clips written to {args.out}")


if __name__ == "__main__":
    main()