"""Measure the scorer the way the 2 GB server would run it: CPU only, fixed threads, optionally 8-bit.

Run as its own process so the memory figures are clean (not mixed with the notebook's own models):
    !python quant_check.py --models {runs}/stage6b/best {runs}/stage7_hubert/best \
        --pitch {runs}/tone23_pitch.joblib --quantize --threads 2 --n 600 --out {runs}/quant_int8.pkl
    !python quant_check.py ... (same, without --quantize) --out {runs}/quant_fp32.pkl   # for comparison

Prints peak memory (what the server must fit) and time per request; saves the raw outputs so a
notebook cell can compare the verdicts with the full-precision results from cell 13.
"""
import argparse
import json
import pickle
import random
import resource
import time


def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--manifests", nargs="+", default=[
        "manifests/syl_test.jsonl", "manifests/good/test_studycli.jsonl", "manifests/good/test_uvm.jsonl"])
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n", type=int, default=600, help="random test clips to score")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"python + nothing loaded: {rss_mb():.0f} MB")
    import torch
    from score import Scorer
    print(f"after importing torch/transformers: {rss_mb():.0f} MB")
    t0 = time.time()
    scorer = Scorer(args.models, device="cpu", pitch_model=args.pitch,
                    quantize=args.quantize, threads=args.threads)
    print(f"models loaded ({'8-bit' if args.quantize else 'full precision'}, {args.threads} threads) "
          f"in {time.time() - t0:.0f} s: {rss_mb():.0f} MB now")

    clips = []
    for m in args.manifests:
        for line in open(m):
            r = json.loads(line)
            clips.append({"audio": r["audio"], "voice": r.get("speaker") or r.get("source"), "pinyin": r["pinyin"]})
    random.seed(0)
    clips = random.sample(clips, min(args.n, len(clips)))

    out, times = {}, []
    for i, c in enumerate(clips, 1):
        wav = Scorer.prepare(c["audio"])
        t = time.time()
        mp = scorer.model_probs(wav)
        vote = scorer.pitch_vote(wav)
        times.append(time.time() - t)
        out[c["audio"]] = {"mp": mp, "vote": vote, "voice": c["voice"], "pinyin": c["pinyin"]}
        if i % 100 == 0:
            print(f"  {i}/{len(clips)}", flush=True)
    pickle.dump(out, open(args.out, "wb"))

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    ms = 1000 * sorted(times)[len(times) // 2]
    p90 = 1000 * sorted(times)[int(0.9 * len(times))]
    print(f"\nPEAK MEMORY: {peak:.0f} MB    (server has ~1,300 MB free)")
    print(f"time per request on this CPU: median {ms:.0f} ms, 90th percentile {p90:.0f} ms")


if __name__ == "__main__":
    main()