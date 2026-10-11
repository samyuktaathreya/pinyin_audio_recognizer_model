"""Build manifests from a folder of isolated-syllable recordings (e.g. Tone Perfect).

The syllable, tone and speaker are read from each file name. The default pattern fits
names like  ma1_FV1_MP3.mp3  (syllable+tone, underscore, speaker). Pass --pattern with
named groups `syl` and `spk` if your files are named differently.

Speakers listed in --test-speakers are held out entirely and written to the test manifest.

Example:
    python prepare_syllables.py --audio-dir tone_perfect --test-speakers MV3 FV3 \
        --out-train manifests/syl_train.jsonl --out-test manifests/syl_test.jsonl
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import soundfile as sf

from pinyin_units import split_syllable

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--pattern", default=r"^(?P<syl>[a-zA-Züv:]+[1-5])_(?P<spk>[A-Za-z0-9]+)")
    p.add_argument("--test-speakers", nargs="*", default=[])
    p.add_argument("--out-train", required=True)
    p.add_argument("--out-test", required=True)
    args = p.parse_args()

    pattern = re.compile(args.pattern)
    held_out = set(args.test_speakers)
    rows = {"train": [], "test": []}
    speakers, unmatched = Counter(), []

    for path in sorted(Path(args.audio_dir).rglob("*")):
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        m = pattern.match(path.name)
        syllable = m.group("syl").lower() if m else None
        if syllable is None or split_syllable(syllable) is None:
            unmatched.append(path.name)
            continue
        speaker = m.group("spk")
        speakers[speaker] += 1
        rows["test" if speaker in held_out else "train"].append(
            {"audio": str(path.resolve()), "duration": round(sf.info(str(path)).duration, 3),
             "pinyin": syllable, "speaker": speaker})

    for split, out in (("train", args.out_train), ("test", args.out_test)):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for row in rows[split]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows[split])} clips -> {out}")

    print("clips per speaker:", dict(speakers))
    missing = held_out - set(speakers)
    if missing:
        print(f"WARNING: test speakers not found in any file name: {sorted(missing)}")
    if unmatched:
        print(f"WARNING: {len(unmatched)} files not understood, e.g. {unmatched[:5]}")


if __name__ == "__main__":
    main()
