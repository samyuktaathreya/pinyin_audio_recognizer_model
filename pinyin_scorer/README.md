# pinyin_scorer

A model that judges how well a Mandarin learner pronounces one pinyin syllable, giving
separate feedback on the **initial**, **final** and **tone**. This file covers the goal,
the code, the data, every training run so far with its results, what the analysis found,
and what is planned next. It is written so that someone (or an AI) new to the project can
pick it up from here.

Last updated: 16 October 2026.

---

## 0. Where things stand (read this first)

The app should use **`score.py` with both models, the forgiveness rules and the pitch vote**:

```python
scorer = Scorer([f"{runs}/stage6b/best", f"{runs}/stage7_hubert/best"],
                pitch_model=f"{runs}/tone23_pitch.joblib")
result = scorer.score("attempt.wav", "zhuang4")
```

| Both web voices (studycli + uvm, never trained on) | old score.py (6b alone) | **current score.py** |
|---|---|---|
| **False alarms** (correct native clip flagged wrong) | 7.5% | **1.9%** |
| All-3 top-guess accuracy | 85.1% | 91.6% |
| Final accuracy (top guess, forgiveness applied) | 93.8% | 98.0% |
| Tone accuracy | 92.3% | 94.8% |

False alarms per voice: studycli 2.4%, uvm 1.5%, FV3 1.0%, MV3 0.5%; about **1.3%** over all
6,151 test clips. Speed: **44 ms per call** (two models + pitch) on a Colab T4; not yet measured
on the server's hardware.

Remaining weak spots: studycli's tone 2 is still heard as tone 3 on ~7% of its clips; uvm's
remaining errors are mostly initials (finals and tones are both at 97.2%). Real learners will do
worse than native speakers, so real false alarms will be higher than 1.9%.

---

## 1. Goal

**Product:** a language-learning app. The learner sees a syllable (e.g. `zhuang4`), says it
into the microphone, and gets feedback like a teacher's: "your initial sounded like *j*, not
*zh*", "your tone came out as 3, not 2".

**The question the model answers:** would a native speaker understand which syllable the
learner meant? It should be forgiving of an accent but catch real mistakes.

**Requirements:**
- **False alarms are the worst failure.** Telling a learner who said it correctly that they
  were wrong is more damaging than missing a mistake. **This is the main number to judge by.**
- **Fast:** under about a second per request. (Azure's pronunciation API was too slow.)
- **Runs on a server** behind an endpoint, so it works in every browser.

**Why a custom model:** existing speech recognisers are built for sentences, output
characters, and use a language model to "fix" what they hear, which hides exactly the
errors a learner makes. Pitch tracking alone handles tones but not consonants or vowels.

**Core idea:** the app always knows the target syllable, so the model does *verification*,
not open-ended recognition ("goodness of pronunciation").

---

## 2. How the model works

```
16 kHz audio ─► encoder (wav2vec2 or HuBERT, ~95M parameters)
             ─► 13 hidden layers ─► learned weighted mix of the layers (one mix per head)
             ─► LayerNorm + dropout ─► three linear CTC heads: initial, final, tone
```

- **Encoder:** `TencentGameMate/chinese-wav2vec2-base` (original) or
  `TencentGameMate/chinese-hubert-base` (newer runs). Both are pretrained without labels on
  10,000 hours of Mandarin. The convolutional front end stays frozen; the 12 transformer
  layers are fine-tuned.
- **Layer mix:** the heads read a softmax-weighted average of all hidden layers, not just
  the last one (reading only the last layer failed completely). With `--per-head-mix`,
  each head learns its own mix.
- **Heads and labels** (`pinyin_units.py`):
  - initial: 21 initials plus `_` for "no initial" (`an`, `wo`, `yi`)
  - final: 39 finals written by sound: `ü` as `v`; `zi/ci/si` vowel as `ii`;
    `zhi/chi/shi/ri` vowel as `iii`; `yi`, `wu`, `yu`, `you`, `wei`, `wen` undone to
    `i`, `u`, `v`, `iu`, `ui`, `un`
  - tone: 1 to 4, plus 5 for neutral
  - every head has a CTC blank at id 0
- **Input handling:** mono, resampled to 16 kHz, loudness-normalised to −20 dBFS, padded to
  at least 0.5 s.

### How a clip is scored (`score.py`)
1. **Each model** computes, per part, the probability of every possible single unit (summing
   over all alignments where exactly that one unit is spoken, with silence around it), plus
   silence, normalised to add up to 1.
2. **The models are combined** by averaging those probabilities (currently 6b + 7).
3. **Pitch re-splits tone 2 vs tone 3** (`pitch_tone.py`, weight `PITCH_WEIGHT = 0.3`). The
   clip's pitch contour is measured with Praat, in semitones relative to the start of the
   syllable (no calibration); a small classifier trained on the training voices turns its
   shape into log-odds of tone 2 over tone 3. The combined tone-2 + tone-3 probability is kept
   and only its split changes: `logit = 0.7 × model's log(p2/p3) + 0.3 × pitch vote`. Skipped
   if no pitch is found (0.2% of clips).
4. **Forgiveness** (`FORGIVE`): the probability of an accepted partner is added to the
   target's. See section 8 for the list.
5. **Decision:**
   - **score** = the target's probability (plus forgiven partners)
   - **heard** = the unit with the highest probability (shown to the learner as "sounded like")
   - **correct** = score ≥ `DEFAULT_THRESHOLD` (**0.05**)
   - **forgiven** = `True` when the top guess was an accepted partner. The app should then not
     say "sounded like *ang*" on an answer marked correct.

Steps 2–5 are plain functions in `score.py` (`combine`, `apply_pitch`, `credited`, `judge`),
so evaluation notebooks call exactly the code the app runs.

### Ways of measuring
- **False-alarm rate**: the share of *correct native* clips where at least one part is flagged
  wrong. **This is what learners experience, and the main gate.**
- **Top-guess accuracy** (`eval_sources.py`, cell 13): a part is right if the top guess is the
  target (or a forgiven partner). Lower than "not flagged", because a target with ≥ 5% that
  isn't the top guess still passes.
- **Swapped-target misses**: native clips of X scored as if the learner were attempting Y. The
  share that passes is how often a real X-for-Y mistake would slip through. Optimistic, since
  real learner mistakes are less clean.
- **Shift test** (`shift_test.py`): robustness to voices unlike the training voices, by
  warping test clips. See section 7.
- **Strict transcription** (`train.py`'s epoch lines): noisy; only for watching training.

---

## 3. Files

| File | What it does |
|---|---|
| `pinyin_units.py` | Splits syllables into parts, defines the three vocabularies, hanzi→pinyin with tone sandhi (for AISHELL) |
| `model.py` | `PinyinCTC`: encoder (loaded with `AutoModel`, so wav2vec2 or HuBERT), layer mix (shared or per head), three heads; CTC loss, greedy decoding, `save()`/`load()` |
| `data.py` | Audio loading, noise augmentation (`augment_audio`), voice/speed augmentation (`change_voice`, `praat_voice`), dataset, collator, duration-bucketed batching |
| `train.py` | Training and `--eval-only` evaluation. Saves `best/` and `last/` every epoch, appends to `log.jsonl` |
| `score.py` | **What the app calls.** `Scorer(model_dirs, pitch_model=...)`; `FORGIVE`, `PITCH_WEIGHT`, `DEFAULT_THRESHOLD`; pure functions `combine`, `apply_pitch`, `credited`, `judge` |
| `pitch_tone.py` | Pitch contour (two-pass Praat), shape measures, and the saved tone 2/3 classifier wrapper (`PitchTone`) |
| `shift_test.py` | Voice-shift robustness test: warps clips in memory with Praat (resonances only, pitch only, both) and scores them |
| `eval_sources.py` | Top-guess accuracy per manifest, the most common mistakes, failures to CSV |
| `dump_probs.py` | Writes every clip's full log-probabilities (every unit, every head) to one CSV |
| `prepare_aishell.py` | AISHELL-1 sentences → manifest (hours-limited random subset) |
| `prepare_syllables.py` | Tone Perfect mp3s → manifests, split by speaker |
| `prepare_folders.py` | One folder per web source → manifests; holds out whole sources for testing |
| `find_duplicates.py`, `audit_clips.py`, `check_augment.py`, `diagnose.py` | Data-quality and debugging tools |

Server dependencies beyond the training ones: `praat-parselmouth`, `scikit-learn`, `joblib`
(for the pitch vote). Load the `.joblib` with the same scikit-learn version it was saved with.

### `train.py` options added recently
| Flag | What it does |
|---|---|
| `--voice-augment` | On 80% of single-syllable training clips: shift the voice (pitch and vowel resonances together) by up to ±2 semitones and the speed by 0.85× to 1.2×. Needs `librosa` |
| `--praat-augment` | Instead of `--voice-augment`: resonances ×0.88–1.13 and pitch ±4 semitones **independently**, via Praat "Change gender", ~10 ms per clip. Tried in stage 8; **not recommended** (section 6) |
| `--per-head-mix` | One layer mix per head instead of a shared one |
| `--freeze-encoder` | Train only the layer mix and heads (encoder probe) |
| `--sentences M.jsonl` + `--sentence-hours H` | Mix a fresh random `H` hours of sentence clips into every epoch |
| `--encoder NAME` | Which pretrained encoder to start from. **Ignored when `--init-from` is given** |

### Manifest format
One JSON object per line:
```json
{"audio": "/content/std/standard/hotoo/a1.wav", "duration": 0.84, "pinyin": "a1", "speaker": "hotoo"}
```
Paths are absolute Colab paths, so manifests are rebuilt every session.

### Saved model folder
```
stage6b/best/
  encoder/      Hugging Face encoder weights + config (config.json "model_type" says wav2vec2 or hubert)
  top.pt        layer mix, LayerNorm, the 3 heads
  units.json    the three vocabularies
```

---

## 4. Data

| Name | What it is | Use |
|---|---|---|
| AISHELL-1 | 15 h random subset of read Mandarin sentences, ~400 speakers (`data_aishell.tgz` on Drive) | Stage 1; mixed into stage 7 |
| Tone Perfect (MSU) | Every syllable in tones 1–4, 6 studio speakers, ~1,640 clips each (`tone_perfect.zip`) | Train: FV1, FV2, MV1, MV2. Test: **FV3, MV3** |
| `standard` web sources | Raw recordings of every syllable, one speaker per source, ~1,600 clips each (`standard.zip`) | See below |
| `processed` | Older cut-up versions of the web sources. **Do not use** | Stages 3–4 only |

**Web sources kept after listening:** `hotoo`, `mandarintemple`, `misc`, `studycli`, `uvm`, `yabla`.
- Train: hotoo, mandarintemple, misc, yabla (`manifests/good/train.jsonl`)
- Test: **studycli** (one speaker, quiet), **uvm** (one speaker, background noise, weaker low
  frequencies, see section 7)
- Skipped: youtube_man, sreading, losttheory, davinfifield, digmandarin, gitcode, hugolpz, yoyochinese
- Known duplicates: gitcode is a copy of sreading; losttheory contains uvm's recordings;
  uvm and shikangkai are the same speaker.
- `misc` has few usable `eng`/`ang` clips (most formant tracks failed); leave it out of
  per-voice conclusions.

**The test that matters:** studycli + uvm together (2,872 clips). Caveats: only two voices,
so 1–2 points between models is not meaningful; and **the four held-out voices have now been
used to choose `PITCH_WEIGHT` and the forgiveness rules**, so they are slightly less clean as a
test than before. A fresh test set (e.g. syllables cut from unseen AISHELL speakers, or real
learners) is needed for an unbiased number.

---

## 5. Colab setup

Drive layout:
```
MyDrive/
  tone_perfect.zip  standard.zip  processed.zip  data_aishell.tgz
  pinyin_runs/stage1 … stage8_praat/{best, last, log.jsonl}
  pinyin_runs/stage2_probs.csv  stage6b_probs.csv  stage7_hubert_probs.csv  stage8_praat_probs.csv
  pinyin_runs/tone23_pitch.joblib         <- pitch classifier used by score.py
  pinyin_runs/pitch_tone23.csv            <- pitch contours of all tone 2/3 clips (cell 10)
  pinyin_runs/formant_raw.csv  formant_ceiling.csv  formant_feats.csv   (formant analysis)
  pinyin_runs/eval_cache_6b_7_pitch.pkl   <- raw score.py outputs for every test clip (cell 13)
  pinyin_runs/shift_6b.csv  shift_8.csv (+ _summary.csv)                (shift tests)
```
Code workflow: edit on the Mac → commit and push → pull in the notebook. **Forgetting to
pull is the most common cause of "unrecognized arguments" errors.** After pulling a changed
`.py` file into a running notebook, **reload it** (`importlib.reload(score)`) or restart the
kernel; otherwise the old version stays in memory.

### Session setup (every session, T4 runtime)
```python
REPO_URL = "https://github.com/YOUR_USER/YOUR_REPO.git"   # <- your repo

from google.colab import drive
drive.mount('/content/drive')
%cd /content
!git clone -q {REPO_URL} repo 2>/dev/null || git -C repo pull -q
%cd /content/repo/pinyin_scorer
%pip install -q pypinyin librosa praat-parselmouth

!unzip -q -o /content/drive/MyDrive/tone_perfect.zip -d /content/tone_perfect
!unzip -q -o /content/drive/MyDrive/standard.zip -d /content/std

!python prepare_syllables.py --audio-dir /content/tone_perfect --test-speakers MV3 FV3 \
    --out-train manifests/syl_train.jsonl --out-test manifests/syl_test.jsonl
!python prepare_folders.py --root /content/std/standard --out-dir manifests/good \
    --test-sources studycli uvm --no-augment-sources uvm \
    --skip-sources youtube_man sreading losttheory davinfifield digmandarin gitcode hugolpz yoyochinese

import os, glob, torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - switch the runtime to T4")
print("forgiveness + pitch in score.py:", "PITCH_WEIGHT" in open("score.py").read())
print("Tone Perfect audio:", os.path.exists("/content/tone_perfect"))
print("web audio:", os.path.exists("/content/std/standard/uvm"))
for m in ["manifests/syl_train.jsonl", "manifests/syl_test.jsonl", "manifests/good/train.jsonl"] + sorted(glob.glob("manifests/good/test_*.jsonl")):
    print(f"{m}: {sum(1 for _ in open(m))} clips")
```
After rebuilding manifests, re-apply any hand-made removals of clips confirmed bad by listening.

### AISHELL setup (only when a run uses sentences)
```python
import os, glob
if not os.path.exists("/content/data_aishell/wav/train"):
    !tar -xzf /content/drive/MyDrive/data_aishell.tgz -C /content
    !cd /content/data_aishell/wav && for f in *.tar.gz; do tar -xzf "$f" && rm "$f"; done
print("speakers unpacked:", len(glob.glob("/content/data_aishell/wav/*/S*")))
!python prepare_aishell.py --aishell-dir /content/data_aishell --split train --hours 15 --out manifests/aishell_train.jsonl
!python prepare_aishell.py --aishell-dir /content/data_aishell --split dev --hours 1 --out manifests/aishell_dev.jsonl
```

---

## 6. Training history

### Early stages (wav2vec2 encoder)
| Stage | Start | Data | Notes | Outcome |
|---|---|---|---|---|
| 1 | pretrained | AISHELL 15 h, 11 epochs | sentence errors 1.4 / 2.4 / 3.1% (initial/final/tone) | Good on sentences, poor on isolated syllables |
| 2 | stage 1 | 4 Tone Perfect speakers, 15 epochs, lr 2e-5/2e-4, no noise | no voice or noise augmentation; used as the "un-augmented" model in tests | Long the best model |
| 3 | stage 1 | Tone Perfect + 11 `processed` sources, strong noise | unstable, cut-off and duplicate clips | Worse. Abandoned |
| 4 | stage 2 | same `processed` data, gentler | stopped when data problems were found | — |
| 5 | stage 2 | Tone Perfect + 8 `standard` sources (4 later dropped as bad), gentle noise | | No better than stage 2 |

### Recent runs, all scored on the same test clips (top-guess, no forgiveness)
| Model | How it was made | Tone Perfect FV3+MV3 | studycli | uvm | **Both web voices** |
|---|---|---|---|---|---|
| stage2 | see above | 93.5% | 75.2% | 86.8% | 80.4% |
| stage6a ("run A") | stage 2 + Tone Perfect + 4 kept web sources, 6 epochs, lr 1.5e-5/1.5e-4, `--augment` | 95.0% | 81.0% | 86.4% | 83.4% |
| stage6b ("run B") | run A + `--voice-augment --per-head-mix` | 94.8% | 85.4% | 84.8% | 85.1% |
| stage7_hubert | HuBERT stage 1, then syllables with 3 h/epoch of AISHELL mixed in, 8 epochs, lr 2e-5/2e-4, `--augment --voice-augment --per-head-mix` | 93.7% | 84.1% | 88.7% | 86.2% |
| **6b + 7 combined** | average the two models' probabilities per clip | 96.1% | 86.9% | 89.8% | **88.2%** |
| stage8_praat | run B's recipe with `--praat-augment` instead of `--voice-augment` | — | 79.6% | 86.8% | 82.9% (false alarms 9.0%) |

**Stage 8 is not used.** It was slightly more robust to artificially warped voices (shift-test
gap 5.3 → 4.9 points) and better on uvm's tones (93.8 → 98.5%), but studycli's tone 2 heard as
3 rose from 140 to 209 clips, its real `eng` errors didn't improve, and it accepted `eng` for an
`ang` target more often (23.6% → 32.5%). Adding it to 6b + 7 made the combination worse.
Shifting pitch while keeping the tone label seems to push the model toward contour shape, which
hurts studycli's unusual tone 2. Resonance shifts near ×1.13–1.15 already make `eng` sound
ambiguous (checked by ear), so the top of that range taught near-`ang` audio labelled `eng`.

Commands used:
```python
runs = "/content/drive/MyDrive/pinyin_runs"
# run B
!python train.py --init-from {runs}/stage2/best \
    --train manifests/syl_train.jsonl manifests/good/train.jsonl \
    --dev manifests/syl_test.jsonl --out {runs}/stage6b \
    --epochs 6 --lr 1.5e-5 --head-lr 1.5e-4 --warmup-steps 200 --augment --workers 2 \
    --voice-augment --per-head-mix
# HuBERT stage 1 (needs AISHELL setup)
!python train.py --encoder TencentGameMate/chinese-hubert-base \
    --train manifests/aishell_train.jsonl --dev manifests/aishell_dev.jsonl \
    --out {runs}/stage1_hubert --epochs 11 --per-head-mix --workers 2
# stage 7 (needs AISHELL setup)
!python train.py --init-from {runs}/stage1_hubert/best \
    --train manifests/syl_train.jsonl manifests/good/train.jsonl \
    --sentences manifests/aishell_train.jsonl --sentence-hours 3 \
    --dev manifests/syl_test.jsonl --out {runs}/stage7_hubert \
    --epochs 8 --lr 2e-5 --head-lr 2e-4 --warmup-steps 200 \
    --augment --voice-augment --per-head-mix --workers 2
```

### Encoder comparison (frozen probe)
| Encoder | Tone Perfect | studycli | uvm | Both web voices | Web finals | Web tones |
|---|---|---|---|---|---|---|
| chinese-wav2vec2-base | 83.7% | 79.8% | 63.8% | 72.6% | 79.4% | 93.1% |
| chinese-hubert-base | 88.3% | 81.9% | 72.4% | 77.6% | 93.1% | 84.7% |
| chinese-hubert-large | 93.1% | 68.6% | 70.5% | 69.5% | 82.6% | 91.4% |

HuBERT large is worse on web voices and slower; ruled out. Whisper was not tried.

---

## 7. What the analysis found

### False alarms and the threshold rule
| Set-up, both web voices | False alarms |
|---|---|
| stage2, target ≥ 0.5 (old rule) | 20.3% |
| stage6b, target ≥ 0.05 | 7.5% |
| 6b + 7 | 4.0% |
| 6b + 7 + forgiveness | 3.3% |
| **6b + 7 + forgiveness + pitch (current)** | **1.9%** |

Thresholds don't transfer evenly: studio voices get far fewer false alarms than web voices.

### The `eng`/`ang` problem (resolved by forgiving it)
The model heard `eng` as `ang` far more than the reverse (studycli 13 and uvm 25 of ~70 `eng`
clips with 6b + 7; FV3 0, MV3 1). What was established:
- **The model judges vowels by absolute frequency, not relative to the speaker.** Warping
  clips' resonances by 10% (shift test) costs 25–30 points on `eng`, `ong`, `ou`; raising
  breaks mid/back vowels (`eng`→`ang`), lowering breaks open vowels and nasal endings
  (`ang`→`ao`, `eng`→`ou`, `in`). Lowering only the pitch also breaks `eng` (88 → 63%), so the
  model reads resonances relative to pitch.
- **The channel matters.** Giving clean voices uvm's average spectrum doubles studycli's
  `eng`→`ang` errors (13 → 25 with 6b + 7; 30 → 51 with un-augmented stage 2). uvm has
  4–8 dB less energy at 100–300 Hz than the other voices, a dip at 1.2–2 kHz and a bump at
  2.5–3.5 kHz.
- **The cue is in the vowel and audible, but the model can't read it on unfamiliar voices.**
  With the nasal cut off (checked by ear: no *ng* left, `e` vs `a` clear to a listener), both
  6b + 7 and stage 2 hear most of studycli's and uvm's misheard `eng` vowels as `a`; FV3 and MV3
  pass the same test.
- **Formant measurements don't fix it.** F1/F2 track well in the vowel (20–50% of the voiced
  region; the nasal starts around 60%). F3 needed a per-clip choice of formant ceiling
  (4,500–6,500 Hz, steadiest track) to stop flipping between bands. Even then F1/F3 separated
  `eng`/`ang` across speakers no better than raw Hz (studycli 72%). studycli's misheard `eng`s
  are genuinely more open (F1 ~850–950 Hz, at the top of its `eng` range); some are ambiguous
  by ear (`feng2`). uvm's misheard `eng`s have normal `eng` formants, so their errors aren't
  about vowel quality. A rescorer combining formants with the model made things much worse
  (fixed 6, broke 170).
- **Decision:** `eng` heard as `ang` (and `en` heard as `an`) is forgiven. See section 8.

### Tone 2 vs tone 3 (partly fixed with pitch)
- studycli's tone 2 dips only ~1.5 semitones, stays low for ~60% of the syllable, then rises
  ~10 semitones; its tone 3 dips ~6. The models heard 154 of its tone 2s as tone 3 with
  confidence (median p(tone 2) 3.6%).
- **Pitch separates them:** a classifier on six contour-shape measures (dip depth, dip position,
  rise, end height, height halfway, creak), trained only on the 8 training voices, put 145–151
  of the 154 right. On its own it also broke correct tone 3s on MV3 and uvm, whose tone 3s dip
  only 3–5 semitones and end above their start. Three training voices (FV2, mandarintemple,
  yabla) show 8–12 semitone "dips" in tone 3 that are creak tracked at half pitch.
- **So pitch only nudges the model** (weight 0.3). Weight sweep, 6b + 7 + forgiveness, web voices:

| Pitch weight | All-3 | False alarms | Tone 3 accepted for tone 2 |
|---|---|---|---|
| 0 | 90.3% | 3.3% | 6.9% |
| 0.2 | 91.2% | 2.2% | 7.9% |
| **0.3** | **91.6%** | **1.9%** | **8.8%** |
| 0.4 | 92.1% | 1.6% | 11.1% |
| 0.5 | 92.3% | 1.3% | 13.9% |

  Capping creak dips, dropping rise/end measures, and a plain "dip deeper than 1.8 semitones =
  tone 3" rule were all tried and were no better.

### Other findings
- **Rare finals are barely learned:** `yo` (final `io`) fails every time, `weng` (`ueng`) about
  half the time, bare `o` often heard as `ou`. Consider leaving `yo` out of drills.
- **hotoo and misc** do badly on `zhi/chi/shi/ri/zi/ci/si` (with stage 2, 41 of 56 clips wrong).
  Listening was planned; the result isn't recorded here.
- **Loudness normalisation** made no difference (the encoder normalises internally).
- **Noise augmentation must stay gentle**; loud noise and low rumble damaged tones and buried
  consonants. The gentle noise in 6b/7 made them *less* channel-sensitive than stage 2.

---

## 8. Decisions made

- **`score.py` combines 6b + 7** (average of probabilities per part).
- **Threshold rule:** a part is correct when its credited probability is at least
  `DEFAULT_THRESHOLD = 0.05`. "heard" still shows the top guess.
- **Forgiveness (`FORGIVE` in `score.py`)**, all on the final:
  - `n`/`ng` swaps, both ways: `an`↔`ang`, `en`↔`eng`, `in`↔`ing`, `ian`↔`iang`, `uan`↔`uang`
  - `eng` heard as `ang`, and `en` heard as `an`, **one way only** (an `ang` heard as `eng` is
    still flagged)
  - Cost (6b + 7, native clips): `ang` said for `eng`, `an` said for `en`, and `in`/`ing` swaps
    now always pass. Gain: web all-3 accuracy 88.2% → 90.3%, false alarms 4.0% → 3.3%.
- **Pitch vote on tone 2 vs 3 at `PITCH_WEIGHT = 0.3`**, classifier `tone23_pitch.joblib`
  (trained on the 8 training voices, notebook cell 12).
- **Stage 8 / `--praat-augment` not adopted** (section 6).
- **Speed:** not optimised; 44 ms per call on a T4 is well within budget. Check on the server.
- `pinyin_units.py` is unchanged; changing the vocabulary would break loading saved models.

---

## 9. Next steps (in order)

1. **Creaky voice counts as tone 3** (in progress): when the target is tone 3 and the clip is
   creaky in the middle of the syllable, accept the tone. Measure first how often tones 1, 2 and
   4 are creaky too, since that is the cost.
2. **Measure speed on the server hardware** (CPU-only could be several times slower than 44 ms).
3. **Record real learners through the app's browser pipeline** and have a native listener judge
   them. This gives the real false-alarm rate, and checks compression and noise suppression.
4. **Fresh test set:** cut syllables out of unseen AISHELL speakers (CTC forced alignment with
   the current model; prefer long, phrase-final syllables). Needed because the four held-out
   voices have been used for tuning. The same pipeline gives more training voices.
5. **Remaining errors:** studycli tone 2/3 (~7% of its clips): feed the pitch contour into the
   tone head and retrain, ideally with more voices. uvm: look at which initials it misses.
6. **Channel augmentation** (random EQ curves, low cuts, band limits) if real learner recordings
   show channel-related errors.

Ideas tried and set aside: TTS-generated training clips (unreliable for single syllables);
HuBERT large; `--praat-augment` voice augmentation (stage 8); formant features and a
formant + model rescorer for `eng`/`ang`; judging the vowel against its own nasal; a vowel-only
(a vs e) head (the nasal ending turned out to *help* the model recognise `eng`).

---

## 10. Useful cells

**Evaluate the current score.py on every test clip** (cell 13; ~8 min the first time, cached after)
— prints all-3 accuracy, false alarms, tone and final accuracy per voice for: old 6b alone,
6b + 7, + forgiveness, + pitch; then swapped-target misses; then checks `score()` agrees and times it.

**Retrain the pitch classifier** (cell 12) after changing `pitch_tone.py`'s features:
reads `pitch_tone23.csv`, trains on the training voices, saves `tone23_pitch.joblib`.

**Score one recording**
```python
!python score.py --model {runs}/stage6b/best {runs}/stage7_hubert/best \
    --pitch-model {runs}/tone23_pitch.joblib --audio attempt.wav --target zhuang4
```

**Shift test** (robustness to unfamiliar voices; keep running it for every new model)
```python
!python shift_test.py --model {runs}/stage6b/best \
    --manifests manifests/syl_test.jsonl manifests/good/test_*.jsonl \
    --sources FV3 MV3 studycli --out {runs}/shift_6b.csv
```
Compare each condition with `praat_1.00`, not `orig`. Judge on the ±10% columns; ±15% resonance
shifts make `eng` genuinely ambiguous. Give two `--model` folders to test the combination.

**Compare models on the test voices**
```python
for name in ["stage6b", "stage7_hubert"]:
    !python eval_sources.py --model {runs}/{name}/best \
        --manifests manifests/syl_test.jsonl manifests/good/test_*.jsonl --out {name}_good.csv
```

**Dump probabilities for analysis**
```python
!python dump_probs.py --model {runs}/stage7_hubert/best \
    --manifests manifests/syl_test.jsonl manifests/good/test_*.jsonl \
    --out {runs}/stage7_hubert_probs.csv
```
One row per clip: `source, audio, target, seconds, target_initial, target_final, target_tone`,
then one column per unit per head (`final:eng`, `tone:2`, ...) holding the log-probability,
plus `<silence>` per head. `stage2_probs.csv` covers all 8 voices; the others cover only the 4
test voices.

**Listen to a kind of mistake** (needs the audio unzipped in this session)
```python
import pandas as pd
from IPython.display import Audio, display

CSV = "/content/drive/MyDrive/pinyin_runs/stage6b_probs.csv"
SOURCE = "uvm"       # or "studycli", "FV3", "MV3", None for all
PART   = "final"     # "initial", "final" or "tone"
TARGET = "eng"       # tones as text: "2"
HEARD  = "ang"       # or None for any wrong answer
LIMIT  = 30

df = pd.read_csv(CSV, dtype=str, keep_default_na=False)
cols = [c for c in df.columns if c.startswith(PART + ":") and "silence" not in c]
probs = df[cols].astype(float)
df["heard"] = probs.idxmax(axis=1).str.split(":").str[1]
df["p_target"] = [2.718281828 ** probs.at[i, f"{PART}:{t}"] for i, t in zip(df.index, df["target_" + PART])]
m = (df["target_" + PART] == TARGET) & (df.heard != TARGET)
if HEARD:  m &= df.heard == HEARD
if SOURCE: m &= df.source == SOURCE
hits = df[m].sort_values("p_target")
print(f"{len(hits)} clips labelled {TARGET} where the model heard {HEARD or 'something else'}\n")
for _, r in hits.head(LIMIT).iterrows():
    print(f"{r.source:9} {r.target:8} heard {r.heard:5} target's probability {r.p_target:.1%}")
    display(Audio(filename=r.audio))
```

---

## 11. Things that bit us

- **Heads must read all layers**; last-layer-only output was all blanks.
- **Data quality beats quantity.** Always check new sources for duplicates
  (`find_duplicates.py`), cut-off clips (`audit_clips.py`), then listen.
- **`--init-from` overrides `--encoder`.** To try a new encoder, train its own stage 1 first.
- **Pull before running** new code, or new flags show up as "unrecognized arguments".
- **Reload after pulling:** a notebook keeps the old copy of an imported `.py` file
  (`ImportError: cannot import name ...`). Use `importlib.reload(module)` or restart the kernel.
- **Run display code inside the notebook**, not with `!python file.py`: in a separate process,
  plots and audio players print as text (`Figure(1000x360)`, `<IPython.lib.display.Audio object>`).
  Use `%run -i file.py` or paste the cells.
- **Notebook variables get overwritten** by later cells that reuse a name (`d`); rerun the cell
  that defines it if you get an `AttributeError` on a column that should exist.
- **pandas sorts pivot-table columns alphabetically**, so "none" can end up as the last column.
- **A new session wipes the Colab disk.** Rerun setup before anything that reads audio;
  `IPython.display.Audio` given a missing path fails with a confusing
  "rate must be specified" error.
- **Praat formant tracking needs the right ceiling**: a single ceiling chosen from pitch made F3
  flip between bands; measuring at several ceilings and keeping the steadiest fixed it.
- **Praat pitch can lock onto half the pitch in creaky voice** (8–12 semitone "dips" in tone 3);
  track in a range around the clip's own median, as `pitch_tone.py` does.
- **`train.py` prints nothing for the first few minutes** (loading, then scoring the dev set).
- **Don't start a second run while one is going** (GPU memory). `!pkill -f train.py` stops a run.
- **Epoch-to-epoch swings of 2–3 points** are normal; `best/` keeps the best epoch.
- **Disconnect the runtime when done**; an idle T4 still uses compute units.