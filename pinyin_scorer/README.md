# Pinyin pronunciation scorer

One pretrained Mandarin encoder (`TencentGameMate/chinese-wav2vec2-base`) with three
output heads: initial, final, tone. Trained on sentences first, then on isolated syllables.

**Status: written but not run.** Expect a few small fixes on first run.

## Files

| File | Purpose |
|---|---|
| `pinyin_units.py` | Splits `zhuang4` into `zh` / `uang` / `4`; the three vocabularies; hanzi to pinyin |
| `model.py` | Encoder + three heads, CTC loss, save/load |
| `data.py` | Manifest reading, audio loading, batching |
| `prepare_aishell.py` | AISHELL-1 to a sentence manifest |
| `prepare_syllables.py` | Folder of isolated syllables to train/test manifests |
| `train.py` | Training and evaluation |
| `score.py` | Scores one learner recording against a target syllable |

## Run order

```bash
pip install -r requirements.txt

# 1. Data
python prepare_aishell.py --aishell-dir data_aishell --split train --hours 15 --out manifests/aishell_train.jsonl
python prepare_aishell.py --aishell-dir data_aishell --split dev   --hours 1  --out manifests/aishell_dev.jsonl
python prepare_syllables.py --audio-dir tone_perfect --test-speakers MV3 FV3 \
    --out-train manifests/syl_train.jsonl --out-test manifests/syl_test.jsonl

# 2. Stage 1: sentences
python train.py --train manifests/aishell_train.jsonl --dev manifests/aishell_dev.jsonl --out runs/stage1

# 3. Checkpoint A: stage-1 model on held-out isolated syllables (expect mediocre, tone worst)
python train.py --init-from runs/stage1/best --dev manifests/syl_test.jsonl --eval-only

# 4. Stage 2: isolated syllables. Its "epoch N" lines are checkpoint B.
python train.py --init-from runs/stage1/best --train manifests/syl_train.jsonl \
    --dev manifests/syl_test.jsonl --out runs/stage2 --epochs 15 --lr 2e-5 --warmup-steps 100

# 5. Score a recording
python score.py --model runs/stage2/best --audio attempt.wav --target zhuang4
```

The number to compare between A and B is `syllable_all3_acc`: the share of held-out
single-syllable clips where initial, final and tone were all recognised correctly.

## Things to know

- **Out of GPU memory:** lower `--batch-seconds` (try 50) and set `--grad-accum 2`.
- **Tone labels follow speech, not the dictionary.** `prepare_aishell.py` applies tone
  sandhi (3+3 becomes 2+3, plus the changes for 一 and 不). `--no-sandhi` turns it off.
- **Three sounds share the letter i.** The vowel in `zi/ci/si` is labelled `ii`, the one
  in `zhi/chi/shi/ri` is `iii`, and the ordinary one is `i`. `ü` is stored as `v`.
- **File names for isolated syllables.** The default pattern assumes `ma1_FV1_MP3.mp3`.
  If `prepare_syllables.py` reports files it didn't understand, pass your own `--pattern`.
- **The 0.5 threshold in `score.py` is a placeholder.** Tune it on real learner recordings.
- **Single syllables only** in `score.py` for now.
