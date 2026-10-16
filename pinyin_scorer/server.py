"""HTTP endpoint for the pinyin scorer.

    POST /score   multipart form: audio=<file>, target=<pinyin with tone, e.g. zhuang4>
    GET  /health

Run locally:
    MODEL_DIR=./models uvicorn server:app --host 0.0.0.0 --port 8000

MODEL_DIR must contain (8-bit ONNX models from export_onnx.py; no PyTorch needed):
    stage6b/          model_int8.onnx, onnx.json
    stage7_hubert/    model_int8.onnx, onnx.json
    tone23_pitch.joblib

The browser can send whatever MediaRecorder produces (webm/opus, ogg, mp4/aac) or a wav file;
ffmpeg converts it to 16 kHz mono before scoring.
"""
import os
import subprocess
import time

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from onnx_scorer import OnnxScorer
from pinyin_units import split_syllable

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
THREADS = int(os.environ.get("THREADS", "0")) or None      # CPU threads (set to the server's core count)
MAX_BYTES = 2_000_000          # ~2 MB upload limit
MAX_SECONDS = 4.0              # a single syllable; longer audio is rejected
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title="pinyin scorer")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["POST", "GET"],
                   allow_headers=["*"])

scorer = OnnxScorer([os.path.join(MODEL_DIR, "stage6b"), os.path.join(MODEL_DIR, "stage7_hubert")],
                    pitch_model=os.path.join(MODEL_DIR, "tone23_pitch.joblib"), threads=THREADS)


def decode(data: bytes) -> np.ndarray:
    """Any audio format ffmpeg understands -> mono float32 at 16 kHz."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
        input=data, capture_output=True, timeout=20)
    if proc.returncode != 0:
        raise HTTPException(400, f"could not read the audio: {proc.stderr.decode()[:200]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/score")
async def score_endpoint(audio: UploadFile = File(...), target: str = Form(...)):
    target = target.strip().lower()
    if split_syllable(target) is None:
        raise HTTPException(400, f"not a pinyin syllable with tone: {target!r}")
    data = await audio.read()
    if not data:
        raise HTTPException(400, "empty audio")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "audio too large")
    wav = decode(data)
    if len(wav) < 1600:
        raise HTTPException(400, "audio too short")
    if len(wav) > MAX_SECONDS * 16000:
        raise HTTPException(400, f"audio longer than {MAX_SECONDS:.0f} s; send one syllable")
    t0 = time.time()
    result = scorer.score(wav, target)
    result["ms"] = round(1000 * (time.time() - t0))
    return result