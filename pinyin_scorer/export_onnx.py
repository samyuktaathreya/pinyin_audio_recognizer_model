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
    ap.add_argument("--check-manifest", default="manifests/good/test_studycli.jsonl",
                    help="real clips to compare the exported model on (8 are picked at random)")
    ap.add_argument("--keep-fp32", action="store_true", help="also keep the full-precision ONNX file")
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

    # Only the matrix multiplications (the transformer's linear layers) go to 8 bits, like PyTorch's
    # quantize_dynamic(nn.Linear). The convolutional front end that reads the raw waveform stays in full
    # precision: quantizing it too (ONNX Runtime's default) damaged the outputs badly.
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
                     per_channel=True)
    with open(os.path.join(args.out, "onnx.json"), "w") as f:
        json.dump({"file": "model_int8.onnx", "heads": list(HEADS), "do_normalize": do_norm}, f, indent=2)
    print(f"saved {int8}: {os.path.getsize(int8) / 1e6:.0f} MB")

    # check on real clips: original PyTorch vs full-precision ONNX (export correct?) vs 8-bit ONNX
    import random
    import onnxruntime as ort
    paths = [json.loads(l)["audio"] for l in open(args.check_manifest)]
    random.seed(0)
    clips = [load_audio(p) for p in random.sample(paths, min(8, len(paths)))]
    sessions = {"full-precision ONNX": ort.InferenceSession(fp32, providers=["CPUExecutionProvider"]),
                "8-bit ONNX": ort.InferenceSession(int8, providers=["CPUExecutionProvider"])}
    worst = {k: 0.0 for k in sessions}
    for wav in clips:
        feats = fe(wav, sampling_rate=16000, return_tensors="pt")["input_values"]
        with torch.no_grad():
            ref = model(feats)
        x = normalise(wav, do_norm)
        if not np.allclose(x, feats.numpy(), atol=1e-4):
            print("WARNING: audio normalisation differs from the feature extractor's")
        for name, sess in sessions.items():
            got = sess.run(None, {"input_values": x})
            for i, h in enumerate(HEADS):
                a, b = head_probs(ref[h][0].float().numpy(), h), head_probs(got[i][0], h)
                worst[name] = max(worst[name], max(abs(a[u] - b[u]) for u in a))
    for name, w in worst.items():
        print(f"largest probability difference vs the original, {len(clips)} real clips, {name}: {w:.3f}")
    if not args.keep_fp32:
        del sessions
        os.remove(fp32)


if __name__ == "__main__":
    main()