"""Build a training manifest from AISHELL-1: hanzi transcripts -> pinyin, random subset by hours.

Expects the standard layout after extracting the per-speaker archives in wav/:
    data_aishell/transcript/aishell_transcript_v0.8.txt
    data_aishell/wav/train/S0002/BAC009S0002W0122.wav

Example:
    python prepare_aishell.py --aishell-dir data_aishell --split train --hours 15 --out manifests/aishell_train.jsonl
    python prepare_aishell.py --aishell-dir data_aishell --split dev   --hours 1  --out manifests/aishell_dev.jsonl
"""
import argparse
import json
import random
from pathlib import Path

import soundfile as sf

from pinyin_units import hanzi_to_pinyin


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aishell-dir", required=True)
    p.add_argument("--split", default="train", choices=["train", "dev", "test"])
    p.add_argument("--hours", type=float, default=15.0, help="0 = use everything")
    p.add_argument("--out", required=True)
    p.add_argument("--no-sandhi", action="store_true", help="keep dictionary tones")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.aishell_dir)
    transcripts = {}
    with open(root / "transcript" / "aishell_transcript_v0.8.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                transcripts[parts[0]] = parts[1]

    wavs = sorted((root / "wav" / args.split).rglob("*.wav"))
    if not wavs:
        raise SystemExit(f"no wav files under {root / 'wav' / args.split} "
                         "(did you extract the per-speaker .tar.gz files?)")
    random.Random(args.seed).shuffle(wavs)

    budget = args.hours * 3600 if args.hours > 0 else float("inf")
    total, kept, dropped = 0.0, 0, 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        for wav in wavs:
            if total >= budget:
                break
            text = transcripts.get(wav.stem)
            pinyin = hanzi_to_pinyin(text, sandhi=not args.no_sandhi) if text else None
            if pinyin is None:
                dropped += 1
                continue
            duration = sf.info(str(wav)).duration
            out.write(json.dumps({"audio": str(wav.resolve()), "duration": round(duration, 3),
                                  "pinyin": pinyin, "speaker": wav.parent.name, "text": text},
                                 ensure_ascii=False) + "\n")
            total += duration
            kept += 1

    print(f"wrote {kept} clips, {total / 3600:.1f} hours -> {args.out}  (dropped {dropped})")


if __name__ == "__main__":
    main()
