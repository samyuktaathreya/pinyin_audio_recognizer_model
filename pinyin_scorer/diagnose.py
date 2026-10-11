"""Can the model learn at all? Overfits ONE small batch and reports what's going on inside.

A working model memorises 8 clips within a few hundred steps. If it can't, there's a bug,
and the printed numbers show where. About 2-3 minutes on a T4.

    python diagnose.py --manifest manifests/aishell_train.jsonl
    python diagnose.py --manifest manifests/aishell_train.jsonl --amp     # same, with mixed precision
"""
import argparse

import torch
import torch.nn.functional as F

from data import ClipDataset, Collator, read_manifests
from model import PinyinCTC, ctc_losses, greedy_decode
from pinyin_units import HEADS, VOCABS


def grad_norm(params):
    norms = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    return torch.norm(torch.stack(norms)).item() if norms else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--n", type=int, default=8, help="clips in the batch")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PinyinCTC.from_encoder(mask_time_prob=0.0)
    model.encoder.freeze_feature_encoder()
    model.to(device)
    fe = PinyinCTC.load_feature_extractor("TencentGameMate/chinese-wav2vec2-base")

    rows = sorted((r for r in read_manifests([args.manifest]) if r["duration"] >= 2.0),
                  key=lambda r: r["duration"])[:args.n]
    data = ClipDataset(rows)
    batch = Collator(fe)([data[i] for i in range(len(rows))])
    x = batch["input_values"].to(device)
    frames = model.frame_lengths(batch["sample_lengths"])
    labels = {h: batch["labels"][h].to(device) for h in HEADS}
    label_lengths = batch["label_lengths"]
    print(f"{len(rows)} clips, {sum(r['duration'] for r in rows):.1f} s of audio, amp={args.amp}")

    # 1. Are the encoder's features informative? Frames of speech should differ from each other.
    print("\n-- encoder features on clip 0 (before training) --")
    model.eval()
    with torch.no_grad():
        hidden = model.encoder(x[:1], output_hidden_states=True).hidden_states
    n0 = frames[0].item()
    for layer in range(0, len(hidden), 2):
        h = hidden[layer][0, :n0].float()
        unit = F.normalize(h, dim=-1)
        sim = unit @ unit.T
        off_diag = (sim.sum() - sim.diag().sum()) / (n0 * n0 - n0)
        print(f"layer {layer:2d}: size={h.norm(dim=-1).mean():8.2f}  "
              f"variation over time={h.std(0).mean():.4f}  "
              f"avg similarity between frames={off_diag:.3f}")

    # 2. Train on this one batch.
    print("\n-- training on the same batch --")
    model.train()
    enc_params = [q for q in model.encoder.parameters() if q.requires_grad]
    head_params = model.top_parameters()
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr},
                             {"params": head_params, "lr": args.head_lr}], weight_decay=0.0)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)
    for step in range(1, args.steps + 1):
        with torch.autocast(device_type=device.type, enabled=args.amp):
            logits = model(x)
        losses = ctc_losses(logits, frames, labels, label_lengths)
        loss = sum(losses.values())
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        enc_g, head_g = grad_norm(enc_params), grad_norm(head_params)
        torch.nn.utils.clip_grad_norm_(enc_params + head_params, 5.0)
        scaler.step(opt)
        scaler.update()
        if step == 1 or step % 20 == 0:
            blank = logits["tone"].float().softmax(-1)[..., 0].mean().item()
            print(f"step {step:3d}  " + "  ".join(f"{h}={losses[h].item():.3f}" for h in HEADS)
                  + f"  grad enc={enc_g:.3g} heads={head_g:.3g}  blank_prob={blank:.2f}"
                  + (f"  scale={scaler.get_scale():.0f}" if args.amp else ""), flush=True)

    print("\nlayer mix (weight per encoder layer, 0 = bottom):",
          " ".join(f"{w:.2f}" for w in model.layer_mix()))

    # 3. What does it output now?
    print("\n-- predictions after training --")
    model.eval()
    with torch.no_grad():
        decoded = greedy_decode(model(x), frames)
    for b in range(2):
        n = label_lengths[b].item()
        print(f"clip {b}: {rows[b]['pinyin']}")
        for h in HEADS:
            ref = " ".join(VOCABS[h][i] for i in batch["labels"][h][b, :n].tolist())
            hyp = " ".join(VOCABS[h][i] for i in decoded[h][b])
            print(f"   {h:7} target: {ref}\n   {'':7} output: {hyp}")


if __name__ == "__main__":
    main()
