"""Write a few clips plus noisy versions of them as .wav files, to listen to.

    python check_augment.py /path/to/processed        # 5 random clips
    python check_augment.py /path/to/processed 10     # 10 random clips

Each clip becomes four files in ./augment_check: the original and three noisy versions,
made with exactly the same code the training run uses.
"""
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from pathlib import Path
import random
import sys

# Add parent directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

from pinyin_scorer.data import SAMPLE_RATE, augment_audio, load_audio

root = Path(sys.argv[1]).expanduser()
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
out = Path("augment_check")
out.mkdir(exist_ok=True)

files = sorted(f for f in root.rglob("*")
               if f.suffix.lower() in {".wav", ".mp3", ".flac"} and not f.name.startswith("._"))
if not files:
    raise SystemExit(f"no audio files found anywhere under {root.resolve()}")
for f in random.sample(files, min(n, len(files))):
    wav = load_audio(str(f))
    stem = f"{f.parent.name}_{f.stem}"
    sf.write(out / f"{stem}_0_original.wav", wav, SAMPLE_RATE)
    for i in range(1, 4):
        sf.write(out / f"{stem}_{i}_noisy.wav",
                 augment_audio(wav, np.random.default_rng(i)), SAMPLE_RATE)
print(f"wrote {len(list(out.glob('*.wav')))} files to {out.resolve()}")