"""Score folders of isolated-syllable recordings the way the app will, and list the failures.

Unlike `train.py --eval-only` (which makes the model transcribe freely), this tells the model
which syllable was attempted and asks what it heard for each part, exactly like score.py.
That is the number that matters for the app.

    python eval_sources.py --model /content/drive/MyDrive/pinyin_runs/stage2/best \
        --manifests manifests/ext/*.jsonl --out failures.csv

Writes every failure to the CSV (file, target, what was heard) so you can listen to them.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from pinyin_units import HEADS, split_syllable
from score import Scorer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--manifests", nargs="+", required=True)
    p.add_argument("--limit", type=int, default=0, help="max clips per manifest; 0 = all")
    p.add_argument("--out", default="failures.csv")
    args = p.parse_args()

    scorer = Scorer(args.model)
    confusions = {h: Counter() for h in HEADS}
    totals = Counter()
    failures = []

    print(f"{'source':16} {'clips':>6} {'initial':>8} {'final':>8} {'tone':>8} {'all 3':>8}")
    for manifest in args.manifests:
        source = Path(manifest).stem
        rows = [json.loads(line) for line in open(manifest, encoding="utf-8") if line.strip()]
        rows = [r for r in rows if split_syllable(r["pinyin"])]
        if args.limit:
            rows = rows[:args.limit]
        ok = Counter()
        n = 0
        for r in rows:
            try:
                result = scorer.score(r["audio"], r["pinyin"])
            except Exception as e:
                print(f"   could not score {r['audio']}: {e}")
                continue
            n += 1
            parts = result["parts"]
            right = {h: parts[h]["heard_unit"] == parts[h]["target_unit"] for h in HEADS}
            for h in HEADS:
                ok[h] += right[h]
                if not right[h]:
                    confusions[h][(parts[h]["target"], parts[h]["heard"])] += 1
            ok["all"] += all(right.values())
            if not all(right.values()):
                failures.append({
                    "source": source, "audio": r["audio"], "target": r["pinyin"],
                    **{f"heard_{h}": parts[h]["heard"] for h in HEADS},
                    **{f"wrong_{h}": int(not right[h]) for h in HEADS},
                    **{f"target_score_{h}": parts[h]["score"] for h in HEADS},
                })
        if n:
            print(f"{source:16} {n:6d} " + " ".join(
                f"{ok[k] / n:8.1%}" for k in ("initial", "final", "tone", "all")), flush=True)
            totals["n"] += n
            for k in ("initial", "final", "tone", "all"):
                totals[k] += ok[k]

    if totals["n"]:
        n = totals["n"]
        print(f"{'ALL':16} {n:6d} " + " ".join(
            f"{totals[k] / n:8.1%}" for k in ("initial", "final", "tone", "all")))

    print("\nmost common mistakes (target -> heard, count):")
    for h in HEADS:
        top = ", ".join(f"{t}->{hd} ({c})" for (t, hd), c in confusions[h].most_common(10))
        print(f"  {h:7}: {top or 'none'}")

    if failures:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(failures[0]))
            writer.writeheader()
            writer.writerows(failures)
        print(f"\n{len(failures)} failures written to {args.out}")


if __name__ == "__main__":
    main()
