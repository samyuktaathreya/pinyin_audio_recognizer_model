"""Build train/test manifests from folders of isolated syllables, one folder per source.

File names start with the syllable and tone, e.g. a1_01307.wav or lv3_002.wav. Whole folders
listed in --test-sources are held out, so the test measures voices the model has never heard.

    python prepare_folders.py --root /content/ext/processed --out-dir manifests/folders

Writes:
    manifests/folders/train.jsonl          every clip from the training folders
    manifests/folders/test.jsonl           every clip from the held-out folders
    manifests/folders/test_<source>.jsonl  the same, one file per held-out folder
"""
import argparse
import json
import re
from pathlib import Path

import soundfile as sf

from pinyin_units import split_syllable

AUDIO = {".wav", ".mp3", ".flac", ".ogg"}
PATTERN = re.compile(r"^([a-zü:v]+[1-5])(?=_|\.)", re.I)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out-dir", default="manifests/folders")
    p.add_argument("--test-sources", nargs="*",
                   default=["gitcode", "studycli", "uvm", "shikangkai_f"])
    p.add_argument("--skip-sources", nargs="*", default=["youtube_man"],
                   help="folders to ignore entirely (youtube_man has no tone numbers)")
    p.add_argument("--no-augment-sources", nargs="*", default=[],
                   help="folders that already have real background noise; "
                        "training will never add extra noise to these")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split = {"train": [], "test": []}
    per_source = {}

    for src in sorted(d for d in Path(args.root).iterdir() if d.is_dir()):
        if src.name in args.skip_sources or src.name.startswith("__"):
            continue
        rows, bad = [], 0
        for f in sorted(src.rglob("*")):
            if f.suffix.lower() not in AUDIO or f.name.startswith("._"):
                continue
            m = PATTERN.match(f.name)
            syllable = m.group(1).lower() if m else None
            if syllable is None or split_syllable(syllable) is None:
                bad += 1
                continue
            try:
                duration = sf.info(str(f)).duration
            except Exception:
                bad += 1
                continue
            row = {"audio": str(f.resolve()), "duration": round(duration, 3),
                   "pinyin": syllable, "speaker": src.name}
            if src.name in args.no_augment_sources:
                row["augment"] = False
            rows.append(row)
        which = "test" if src.name in args.test_sources else "train"
        split[which] += rows
        if which == "test":
            per_source[src.name] = rows
        note = "  (no added noise)" if src.name in args.no_augment_sources else ""
        print(f"{src.name:16} {which:5} {len(rows):6} clips" + (f"  ({bad} skipped)" if bad else "") + note)

    missing = set(args.test_sources) - set(per_source)
    if missing:
        print(f"WARNING: test sources not found: {sorted(missing)}")

    def write(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write(out / "train.jsonl", split["train"])
    write(out / "test.jsonl", split["test"])
    for name, rows in per_source.items():
        write(out / f"test_{name}.jsonl", rows)
    hours = sum(r["duration"] for r in split["train"]) / 3600
    print(f"\ntrain: {len(split['train'])} clips ({hours:.1f} h)   test: {len(split['test'])} clips")


if __name__ == "__main__":
    main()