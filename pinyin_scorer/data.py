"""Manifest loading, audio loading and batching.

A manifest is a JSON-lines file, one clip per line:
    {"audio": "/path/x.wav", "duration": 3.2, "pinyin": "guang3 zhou1 fang2 chan3"}
Sentences and single syllables use the same format (a syllable is a sentence of length 1).
"""
import json
import random
from math import gcd

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from pinyin_units import HEADS, encode

SAMPLE_RATE = 16000


TARGET_DBFS = -20.0      # loudness every clip is scaled to (typical speech level)
MAX_GAIN_DB = 40.0       # never boost more than this, so near-silent clips don't become loud hiss


def normalize_loudness(wav, target_dbfs=TARGET_DBFS):
    """Scale a clip so its speech has the same loudness as every other clip.

    The model sees raw sample values, so without this a quiet recording (studycli) and
    a loud one look very different to it even when the speech is identical. Training,
    evaluation and the app all go through this, so the model always hears the same level."""
    gain_db = target_dbfs - 10 * np.log10(speech_power(wav))
    wav = wav * 10 ** (min(gain_db, MAX_GAIN_DB) / 20)
    peak = float(np.abs(wav).max()) if len(wav) else 0.0
    if peak > 0.99:                       # loud consonant bursts: scale down instead of clipping
        wav = wav * (0.99 / peak)
    return wav.astype(np.float32)


def load_audio(path, min_seconds=0.5):
    """Mono float32 at 16 kHz, loudness-normalized, padded with silence up to min_seconds."""
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = gcd(sr, SAMPLE_RATE)
        wav = resample_poly(wav, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return pad_audio(normalize_loudness(wav), min_seconds)


def pad_audio(wav, min_seconds=0.5):
    short = int(min_seconds * SAMPLE_RATE) - len(wav)
    if short > 0:
        wav = np.pad(wav, (short // 2, short - short // 2))
    return wav


def read_manifests(paths, max_seconds=16.0):
    rows, skipped = [], 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["labels"] = encode(row["pinyin"])
                if row["labels"] is None or row["duration"] > max_seconds:
                    skipped += 1
                    continue
                rows.append(row)
    if skipped:
        print(f"skipped {skipped} clips (too long or unparseable pinyin)")
    return rows


def speech_power(wav, frame=160):
    """Average power of the loud part of a clip (its top 30% of 10 ms frames).
    Averaging over the whole clip would count silence and make added noise too loud."""
    n = len(wav) // frame
    if n == 0:
        return float(np.mean(wav ** 2)) + 1e-10
    power = (wav[:n * frame].reshape(n, frame) ** 2).mean(axis=1)
    loud = np.sort(power)[-max(1, int(0.3 * n)):]
    return float(loud.mean()) + 1e-10


def augment_audio(wav, rng):
    """Make a clean clip sound more like a learner's recording: extra silence at the
    edges, a different volume, and occasionally faint background hiss.

    The hiss is kept quiet on purpose (25-40 dB below the speech). Louder noise was
    tried and buried quiet consonants like x, s and f, so the model was being trained on
    clips a human couldn't recognise. Real-world noise comes from the noisy recordings
    already in the training data instead."""
    power = speech_power(wav)
    if rng.random() < 0.5:
        pre, post = (rng.uniform(0, 0.4, 2) * SAMPLE_RATE).astype(int)
        wav = np.concatenate([np.zeros(pre, np.float32), wav, np.zeros(post, np.float32)])
    if rng.random() < 0.3:
        noise = rng.standard_normal(len(wav)).astype(np.float32)
        snr_db = rng.uniform(25, 40)
        noise *= np.sqrt(power / 10 ** (snr_db / 10) / (np.mean(noise ** 2) + 1e-10))
        wav = wav + noise
    wav = wav * 10 ** (rng.uniform(-3, 3) / 20)   # small: every clip is loudness-normalized anyway
    return np.clip(wav, -1.0, 1.0).astype(np.float32)


class ClipDataset(Dataset):
    def __init__(self, rows, augment=False):
        self.rows = rows
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        wav = load_audio(row["audio"])
        if self.augment and row.get("augment", True):
            # seeded from torch so each DataLoader worker gets different noise
            wav = augment_audio(wav, np.random.default_rng(torch.randint(0, 2**31, (1,)).item()))
        return {"wav": wav, "labels": row["labels"]}


class Collator:
    def __init__(self, feature_extractor):
        self.fe = feature_extractor

    def __call__(self, items):
        wavs = [it["wav"] for it in items]
        feats = self.fe(wavs, sampling_rate=SAMPLE_RATE, padding=True, return_tensors="pt")
        n = max(len(it["labels"]["tone"]) for it in items)
        labels = {h: torch.zeros(len(items), n, dtype=torch.long) for h in HEADS}
        for b, it in enumerate(items):
            for h in HEADS:
                ids = it["labels"][h]
                labels[h][b, :len(ids)] = torch.tensor(ids)
        return {
            "input_values": feats["input_values"],
            "attention_mask": feats.get("attention_mask"),   # None for base models
            "sample_lengths": torch.tensor([len(w) for w in wavs]),
            "labels": labels,
            "label_lengths": torch.tensor([len(it["labels"]["tone"]) for it in items]),
        }


class DurationBatchSampler:
    """Groups clips of similar length; each batch holds at most `batch_seconds` of padded audio."""

    def __init__(self, rows, batch_seconds=100.0, max_batch_size=64, shuffle=True, seed=0):
        order = sorted(range(len(rows)), key=lambda i: rows[i]["duration"])
        self.batches, batch = [], []
        for i in order:
            longest = max(rows[i]["duration"], 0.5)   # sorted, so the newest is the longest
            if batch and (longest * (len(batch) + 1) > batch_seconds
                          or len(batch) >= max_batch_size):
                self.batches.append(batch)
                batch = []
            batch.append(i)
        if batch:
            self.batches.append(batch)
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.batches)
        return iter(self.batches)