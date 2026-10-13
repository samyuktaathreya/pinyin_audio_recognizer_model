"""Voice-shift robustness test with the factors separated.

For every test clip, makes shifted versions IN MEMORY with Praat's "Change gender"
(through parselmouth) and scores each one. Nothing is written to disk except the CSVs.

Conditions (all keep the clip's length unchanged):
  orig        the untouched clip
  praat_1.00  Praat resynthesis with no change: measures the processing's own damage.
              Compare every other condition against THIS, not against orig.
  F0.85 ...   resonances (formants) only, pitch kept
  P-4 ...     pitch only (semitones), resonances kept
  B0.90 ...   both, by the same factor, like the earlier resampling test but without
              the length change

Usage (Colab, after session setup):
  !python shift_test.py --model {runs}/stage6b/best \
      --manifests manifests/syl_test.jsonl manifests/good/test_*.jsonl \
      --out {runs}/shift_stage6b.csv --check        # 3 clips, checks the scorer output
  !python shift_test.py --model {runs}/stage6b/best \
      --manifests manifests/syl_test.jsonl manifests/good/test_*.jsonl \
      --out {runs}/shift_stage6b.csv
Give two --model folders to test the combined model (probabilities averaged).
"""
import argparse
import collections
import csv
import json
import math
import sys
from multiprocessing import Pool

import numpy as np

SR = 16000
PITCH_FLOOR, PITCH_CEILING = 75.0, 600.0
PARTS = ("initial", "final", "tone")

CONDITIONS = [  # name, formant ratio, pitch shift in semitones (None = untouched clip)
    ("orig", None, None),
    ("praat_1.00", 1.00, 0.0),
    ("F0.85", 0.85, 0.0), ("F0.90", 0.90, 0.0), ("F1.10", 1.10, 0.0), ("F1.15", 1.15, 0.0),
    ("P-4", 1.0, -4.0), ("P-2", 1.0, -2.0), ("P+2", 1.0, 2.0), ("P+4", 1.0, 4.0),
    ("B0.85", 0.85, 12 * math.log2(0.85)), ("B0.90", 0.90, 12 * math.log2(0.90)),
    ("B1.10", 1.10, 12 * math.log2(1.10)), ("B1.15", 1.15, 12 * math.log2(1.15)),
]

DEFAULT_FINALS = ["an", "ang", "en", "eng", "in", "ing", "ong", "ian", "iang",
                  "uan", "uang", "ueng", "iong", "ao", "ou"]

# ---------------------------------------------------------------- pinyin -> parts
# Same conventions as pinyin_units.py (v for ü, ii, iii, y/w forms undone).
# --check and the run itself verify every target against the model's vocabulary.
INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
            "g", "k", "h", "j", "q", "x", "r", "z", "c", "s"]
Y_MAP = {"yi": "i", "ya": "ia", "ye": "ie", "yao": "iao", "you": "iu", "yan": "ian",
         "yin": "in", "yang": "iang", "ying": "ing", "yong": "iong", "yu": "v",
         "yue": "ve", "yuan": "van", "yun": "vn", "yo": "io"}
W_MAP = {"wu": "u", "wa": "ua", "wo": "uo", "wai": "uai", "wei": "ui", "wan": "uan",
         "wen": "un", "wang": "uang", "weng": "ueng"}


def split_pinyin(p):
    p = p.strip().lower().replace("ü", "v").replace("u:", "v")
    tone = p[-1] if p[-1].isdigit() else "5"
    tone = "5" if tone == "0" else tone
    base = p.rstrip("012345")
    if base in Y_MAP:
        return "_", Y_MAP[base], tone
    if base in W_MAP:
        return "_", W_MAP[base], tone
    for ini in INITIALS:
        if base.startswith(ini) and len(base) > len(ini):
            fin = base[len(ini):]
            if ini in ("zh", "ch", "sh", "r") and fin == "i":
                fin = "iii"
            elif ini in ("z", "c", "s") and fin == "i":
                fin = "ii"
            elif ini in ("j", "q", "x") and fin.startswith("u"):
                fin = "v" + fin[1:]
            return ini, fin, tone
    return "_", base, tone


# ---------------------------------------------------------------- audio (workers)
def _worker_init():
    global parselmouth, call, librosa
    import parselmouth
    from parselmouth.praat import call
    import librosa


def make_versions(item):
    """Load one clip and return {condition: float32 array or None}."""
    rec, conds = item
    wav, _ = librosa.load(rec["audio"], sr=SR, mono=True)
    snd = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=SR)
    median = None
    if any(st for _, _, st in conds if st):
        pitch = snd.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
        median = call(pitch, "Get quantile", 0, 0, 0.5, "Hertz")
        if median is None or math.isnan(median):
            median = None
    out = {}
    for name, ratio, st in conds:
        if ratio is None:
            out[name] = wav.astype(np.float32)
            continue
        if st and median is None:  # no pitch found: can't do a pitch shift
            out[name] = None
            continue
        new_median = median * 2 ** (st / 12) if st else 0.0  # 0 = keep original pitch
        shifted = call(snd, "Change gender", PITCH_FLOOR, PITCH_CEILING,
                       float(ratio), float(new_median), 1.0, 1.0)
        out[name] = np.asarray(shifted.values[0], dtype=np.float32)
    return rec, out


# ---------------------------------------------------------------- scoring adapter
def _find_dist(obj):
    """Find a {unit: probability} mapping inside one part of score()'s result."""
    if isinstance(obj, dict):
        if obj and all(isinstance(v, (int, float)) for v in obj.values()):
            return obj
        for k in ("probs", "probabilities", "distribution", "dist", "all"):
            if k in obj:
                return _find_dist(obj[k])
    for k in ("probs", "probabilities", "distribution", "dist"):
        if hasattr(obj, k):
            return _find_dist(getattr(obj, k))
    return None


def get_probs(scorer, wav, syllable):
    """{part: {unit: probability}} for one clip, from Scorer.score()."""
    res = scorer.score(wav, syllable)
    out = {}
    for part in PARTS:
        obj = res.get(part) if isinstance(res, dict) else getattr(res, part, None)
        d = _find_dist(obj) if obj is not None else None
        if d is None:
            sys.exit("Couldn't find the full probability list in Scorer.score()'s result.\n"
                     f"It looks like this:\n{repr(res)[:2000]}\n"
                     "Paste score.py and get_probs() can be fitted to it.")
        d = {str(k).split(":")[-1]: float(v) for k, v in d.items()}
        if any(v < 0 for v in d.values()):  # log-probabilities
            d = {k: math.exp(v) for k, v in d.items()}
        total = sum(d.values()) or 1.0
        out[part] = {k: v / total for k, v in d.items()}
    return out


def ensemble_probs(scorers, wav, syllable):
    all_p = [get_probs(s, wav, syllable) for s in scorers]
    out = {}
    for part in PARTS:
        keys = all_p[0][part].keys()
        out[part] = {k: sum(p[part].get(k, 0.0) for p in all_p) / len(all_p) for k in keys}
    return out


def top_unit(dist):
    units = {k: v for k, v in dist.items() if not k.startswith("<")}  # skip <silence>
    return max(units, key=units.get)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True, help="one folder, or two to combine")
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="per-clip CSV; summary goes to *_summary.csv")
    ap.add_argument("--sources", nargs="*", help="e.g. FV3 MV3 studycli (default: all)")
    ap.add_argument("--finals", nargs="*", default=DEFAULT_FINALS, help="target finals to test")
    ap.add_argument("--conditions", nargs="*", help="subset of condition names")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--check", action="store_true", help="3 clips only, print what's found")
    args = ap.parse_args()

    conds = [c for c in CONDITIONS if not args.conditions or c[0] in args.conditions]
    finals = set(args.finals)

    items = []
    for m in args.manifests:
        for line in open(m):
            rec = json.loads(line)
            rec["source"] = rec.get("source") or rec.get("speaker") or "?"
            if args.sources and rec["source"] not in args.sources:
                continue
            rec["parts"] = dict(zip(PARTS, split_pinyin(rec["pinyin"])))
            if rec["parts"]["final"] in finals:
                items.append((rec, conds))
    if args.check:
        items = items[:3]
    print(f"{len(items)} clips x {len(conds)} conditions", flush=True)

    # Start the audio workers before CUDA is touched in this process.
    pool = Pool(args.workers, initializer=_worker_init)

    from score import Scorer
    try:
        from score import DEFAULT_THRESHOLD
    except ImportError:
        DEFAULT_THRESHOLD = 0.05
    thr = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD
    scorers = [Scorer(m) for m in args.model]

    rows, unknown = [], collections.Counter()
    for i, (rec, versions) in enumerate(pool.imap(make_versions, items, chunksize=4), 1):
        for name, _, _ in conds:
            wav = versions[name]
            if wav is None:
                continue
            probs = ensemble_probs(scorers, wav, rec["pinyin"])
            if any(rec["parts"][p] not in probs[p] for p in PARTS):
                unknown[str(rec["parts"])] += 1
                continue
            row = {"source": rec["source"], "audio": rec["audio"],
                   "target": rec["pinyin"], "condition": name}
            for p in PARTS:
                t = rec["parts"][p]
                row[f"{p}_target"] = t
                row[f"{p}_heard"] = top_unit(probs[p])
                row[f"{p}_p"] = round(probs[p][t], 5)
            rows.append(row)
            if args.check:
                print(f"{rec['source']:9} {rec['pinyin']:8} {name:11} "
                      f"final heard {row['final_heard']:5} p={row['final_p']:.3f}  "
                      f"tone heard {row['tone_heard']} p={row['tone_p']:.3f}")
        if args.check and i == 1:
            fp = ensemble_probs(scorers, versions["orig"] if "orig" in versions
                                else next(v for v in versions.values() if v is not None),
                                rec["pinyin"])
            print("final units found:", sorted(fp["final"]))
            print("tone units found:", sorted(fp["tone"]))
        if i % 100 == 0:
            print(f"  {i}/{len(items)} clips", flush=True)
    pool.close()

    if unknown:
        print("\nWARNING: targets not in the model's vocabulary (skipped):")
        for k, n in unknown.most_common(10):
            print(f"  {k}: {n}")
    if args.check:
        return

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- summary: per source x target final x condition
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["source"], r["final_target"], r["condition"])].append(r)
        groups[("ALL", r["final_target"], r["condition"])].append(r)
    summary = []
    for (src, fin, cond), rs in sorted(groups.items()):
        wrong = collections.Counter(r["final_heard"] for r in rs if r["final_heard"] != fin)
        summary.append({
            "source": src, "final": fin, "condition": cond, "n": len(rs),
            "final_acc": round(100 * sum(r["final_heard"] == fin for r in rs) / len(rs), 1),
            "final_flagged": round(100 * sum(r["final_p"] < thr for r in rs) / len(rs), 1),
            "tone_acc": round(100 * sum(r["tone_heard"] == r["tone_target"] for r in rs) / len(rs), 1),
            "heard_instead": " ".join(f"{u}:{n}" for u, n in wrong.most_common(3)),
        })
    sum_path = args.out.rsplit(".", 1)[0] + "_summary.csv"
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    # ---- printed tables: final accuracy %, rows = final, columns = condition
    names = [c[0] for c in conds]
    by = {(s["source"], s["final"], s["condition"]): s for s in summary}
    for src in sorted({s["source"] for s in summary}, key=lambda x: (x != "ALL", x)):
        print(f"\n=== {src}: final accuracy % (n per cell in brackets)")
        print(f"{'final':6}" + "".join(f"{n:>11}" for n in names))
        for fin in sorted({s["final"] for s in summary if s["source"] == src}):
            cells = []
            for n in names:
                s = by.get((src, fin, n))
                cells.append(f"{s['final_acc']:>6.0f} ({s['n']:>2})" if s else f"{'-':>11}")
            print(f"{fin:6}" + "".join(cells))
    print(f"\nper-clip rows: {args.out}\nsummary (incl. what each final was heard as): {sum_path}")


if __name__ == "__main__":
    main()