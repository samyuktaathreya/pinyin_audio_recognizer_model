"""Find sources that contain the same recordings under different names.

Pass the source folders to compare (each folder = one source). For every syllable two
sources share, it compares the recordings' length (after trimming silence) and the shape
of their loudness over time. Two different speakers rarely match on both; copies of the
same recording match almost every time.

    python find_duplicates.py /path/to/standard/*
    python find_duplicates.py /path/to/standard/* /path/to/processed/gitcode

Prints one row per pair of sources, most suspicious first, and writes every matching
pair of files to duplicate_examples.csv so you can listen to a few.
"""
import argparse
import csv
import itertools
import re
from collections import defaultdict
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000
FRAME = 160                      # 10 ms
AUDIO = {".wav", ".mp3", ".flac", ".ogg"}
PATTERN = re.compile(r"^([a-zü:v]+[1-5])(?=_|\.)", re.I)


def envelope(path):
    """Loudness per 10 ms with silence trimmed off both ends, scaled to a peak of 1."""
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        wav = resample_poly(wav, SR // g, sr // g)
    n = len(wav) // FRAME
    if n < 3:
        return None
    rms = np.sqrt((wav[:n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1))
    active = np.where(rms > 0.05 * rms.max())[0]
    if len(active) < 3:
        return None
    env = rms[active[0]:active[-1] + 1]
    return env / (env.max() + 1e-9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+")
    p.add_argument("--max-frames-apart", type=int, default=2, help="2 = within 20 ms")
    p.add_argument("--min-similarity", type=float, default=0.95)
    p.add_argument("--out", default="duplicate_examples.csv")
    p.add_argument("--summary", default="duplicate_pairs.csv")
    args = p.parse_args()

    sources = [Path(f) for f in args.folders if Path(f).is_dir()]
    names = {s: s.name if [t.name for t in sources].count(s.name) == 1
             else f"{s.parent.name}/{s.name}" for s in sources}
    clips = defaultdict(dict)                    # syllable -> {source: (envelope, path)}
    for src in sources:
        count = 0
        for f in sorted(src.rglob("*")):
            if f.suffix.lower() not in AUDIO or f.name.startswith("._"):
                continue
            m = PATTERN.match(f.name)
            if not m:
                continue
            syllable = m.group(1).lower()
            if names[src] in clips[syllable]:
                continue                         # keep one file per syllable per source
            try:
                env = envelope(str(f))
            except Exception:
                continue
            if env is not None:
                clips[syllable][names[src]] = (env, str(f))
                count += 1
        print(f"read {count:5d} syllables from {names[src]}", flush=True)

    stats = defaultdict(lambda: [0, 0, 0])       # shared, same length, same length + shape
    matches = []
    for syllable, by_source in clips.items():
        for (a, (ea, pa)), (b, (eb, pb)) in itertools.combinations(sorted(by_source.items()), 2):
            s = stats[(a, b)]
            s[0] += 1
            if abs(len(ea) - len(eb)) > args.max_frames_apart:
                continue
            s[1] += 1
            n = min(len(ea), len(eb))
            similarity = float(np.corrcoef(ea[:n], eb[:n])[0, 1])
            if similarity >= args.min_similarity:
                s[2] += 1
                matches.append({"source_a": a, "source_b": b, "syllable": syllable,
                                "similarity": round(similarity, 3), "file_a": pa, "file_b": pb})

    rows = sorted(stats.items(), key=lambda kv: -kv[1][2] / max(kv[1][0], 1))
    print(f"\n{'source A':16} {'source B':16} {'shared':>7} {'same length':>12} {'same recording?':>16}")
    for (a, b), (shared, same_len, same) in rows:
        flag = "   <-- check" if shared and same / shared > 0.3 else ""
        print(f"{a:16} {b:16} {shared:7d} {same_len / shared:12.0%} {same / shared:16.0%}{flag}")

    with open(args.summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_a", "source_b", "shared", "same_recording_fraction"])
        for (a, b), (shared, _, same) in rows:
            writer.writerow([a, b, shared, round(same / shared, 3) if shared else 0])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_a", "source_b", "syllable",
                                               "similarity", "file_a", "file_b"])
        writer.writeheader()
        writer.writerows(sorted(matches, key=lambda m: (m["source_a"], m["source_b"])))
    print(f"\n{len(matches)} matching pairs written to {args.out}")


if __name__ == "__main__":
    main()