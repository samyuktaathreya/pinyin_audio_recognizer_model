"""Export a trained model to ONNX and shrink it to 8 bits, so the server can run it without PyTorch.

Run in Colab (needs torch + onnx + onnxruntime):
    %pip install -q onnx onnxruntime
    !python export_onnx.py --model {runs}/stage6b/best        --out {runs}/onnx/stage6b
    !python export_onnx.py --model {runs}/stage7_hubert/best  --out {runs}/onnx/stage7_hubert

Each output folder holds model_int8.onnx (~4x smaller than the original weights) and onnx.json
(which heads come out in which order, and how audio is normalised). The script finishes by checking
that the 8-bit ONNX model gives nearly the same probabilities as the original on a few clips.
"""
import argparse
import json
import os

import numpy as np
import torch

from model import PinyinCTC
from pinyin_units import HEADS
from scorer_core import head_probs, load_audio


class Wrap(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_values):
        out = self.model(input_values)
        return tuple(out[h] for h in HEADS)


def normalise(wav, do_normalize):
    x = wav.astype(np.float32)
    if do_normalize:                       # same as the Hugging Face feature extractor
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    return x[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check-audio", nargs="*", default=None, help="a few wav files to compare on")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = PinyinCTC.load(args.model).eval()
    fe = PinyinCTC.load_feature_extractor(args.model)
    do_norm = bool(getattr(fe, "do_normalize", True))
    fp32, int8 = os.path.join(args.out, "model_fp32.onnx"), os.path.join(args.out, "model_int8.onnx")

    dummy = torch.randn(1, 16000)
    kw = dict(input_names=["input_values"], output_names=list(HEADS), opset_version=args.opset,
              dynamic_axes={"input_values": {1: "samples"}, **{h: {1: "frames"} for h in HEADS}},
              do_constant_folding=True)
    with torch.no_grad():
        try:
            torch.onnx.export(Wrap(model), (dummy,), fp32, dynamo=False, **kw)
        except TypeError:                  # older torch without the dynamo argument
            torch.onnx.export(Wrap(model), (dummy,), fp32, **kw)

    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
    os.remove(fp32)
    with open(os.path.join(args.out, "onnx.json"), "w") as f:
        json.dump({"file": "model_int8.onnx", "heads": list(HEADS), "do_normalize": do_norm}, f, indent=2)
    print(f"saved {int8}: {os.path.getsize(int8) / 1e6:.0f} MB")

    # check: original PyTorch model vs 8-bit ONNX
    import onnxruntime as ort
    sess = ort.InferenceSession(int8, providers=["CPUExecutionProvider"])
    clips = [load_audio(p) for p in (args.check_audio or [])] or [
        np.random.default_rng(i).standard_normal(16000 + 4000 * i).astype(np.float32) * 0.05 for i in range(3)]
    worst = 0.0
    for wav in clips:
        feats = fe(wav, sampling_rate=16000, return_tensors="pt")["input_values"]
        with torch.no_grad():
            ref = model(feats)
        got = sess.run(None, {"input_values": normalise(wav, do_norm)})
        for i, h in enumerate(HEADS):
            a = head_probs(ref[h][0].float().numpy(), h)
            b = head_probs(got[i][0], h)
            worst = max(worst, max(abs(a[u] - b[u]) for u in a))
    print(f"largest probability difference, original vs 8-bit ONNX, on {len(clips)} clips: {worst:.3f}")


if __name__ == "__main__":
    main()