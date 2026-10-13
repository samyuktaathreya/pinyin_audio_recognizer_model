"""Train the three-head pinyin model.

Stage 1, sentences (teaches the sounds across many speakers):
    python train.py --train manifests/aishell_train.jsonl --dev manifests/aishell_dev.jsonl \
        --out runs/stage1 --epochs 20

Stage 2, isolated syllables (adapts to single-syllable speech):
    python train.py --init-from runs/stage1/best --train manifests/syl_train.jsonl \
        --dev manifests/syl_test.jsonl --out runs/stage2 --epochs 15 --lr 2e-5 --head-lr 2e-4 \
        --warmup-steps 100

Evaluate only:
    python train.py --init-from runs/stage1/best --dev manifests/syl_test.jsonl --eval-only
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import ClipDataset, Collator, DurationBatchSampler, read_manifests
from model import DEFAULT_ENCODER, PinyinCTC, ctc_losses, greedy_decode
from pinyin_units import HEADS


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def sample_hours(rows, hours, seed):
    """A random selection of clips adding up to about `hours` of audio."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    picked, seconds = [], 0.0
    for row in rows:
        if seconds >= hours * 3600:
            break
        picked.append(row)
        seconds += row["duration"]
    return picked


def make_loader(rows, collator, args, shuffle, augment=False, voice_augment=False, seed=None):
    seed = args.seed if seed is None else seed
    sampler = DurationBatchSampler(rows, args.batch_seconds, args.max_batch_size, shuffle, seed)
    return DataLoader(ClipDataset(rows, augment=augment, voice_augment=voice_augment),
                      batch_sampler=sampler, collate_fn=collator, num_workers=args.workers)


def run_model(model, batch, device, use_amp):
    mask = batch["attention_mask"]
    with torch.autocast(device_type=device.type, enabled=use_amp):
        logits = model(batch["input_values"].to(device),
                       None if mask is None else mask.to(device))
    return logits, model.frame_lengths(batch["sample_lengths"])


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    """Error rate per head over all syllables, plus accuracy on single-syllable clips."""
    model.eval()
    errors = {h: 0 for h in HEADS}
    units = 0
    single = {h: 0 for h in HEADS}
    single_all, single_n = 0, 0
    for batch in loader:
        logits, frame_lengths = run_model(model, batch, device, use_amp)
        decoded = greedy_decode(logits, frame_lengths)
        for b, n in enumerate(batch["label_lengths"].tolist()):
            ok = {}
            for h in HEADS:
                ref = batch["labels"][h][b, :n].tolist()
                errors[h] += edit_distance(decoded[h][b], ref)
                ok[h] = decoded[h][b] == ref
            units += n
            if n == 1:
                single_n += 1
                single_all += all(ok.values())
                for h in HEADS:
                    single[h] += ok[h]
    result = {f"{h}_error": errors[h] / max(units, 1) for h in HEADS}
    result["mean_error"] = sum(result.values()) / len(HEADS)
    if single_n:
        for h in HEADS:
            result[f"syllable_{h}_acc"] = single[h] / single_n
        result["syllable_all3_acc"] = single_all / single_n
        result["syllable_clips"] = single_n
    return result


def show(tag, metrics):
    print(tag, "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in metrics.items()), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", nargs="+", default=[])
    p.add_argument("--dev", nargs="+", required=True)
    p.add_argument("--out", default="runs/exp")
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--init-from", help="directory written by an earlier run, e.g. runs/stage1/best")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4, help="learning rate for the encoder")
    p.add_argument("--head-lr", type=float, default=1e-3,
                   help="learning rate for the three output layers")
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--batch-seconds", type=float, default=100.0,
                   help="padded audio per batch; lower this if the GPU runs out of memory")
    p.add_argument("--max-batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--mask-time-prob", type=float, default=0.05)
    p.add_argument("--tone-weight", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=1, help="evaluate and save every N epochs")
    p.add_argument("--no-amp", action="store_true", help="turn off mixed precision")
    p.add_argument("--augment", action="store_true",
                   help="add background noise, volume changes and edge silence to training clips")
    p.add_argument("--voice-augment", action="store_true",
                   help="randomly shift the voice (up to 2 semitones) and the speaking speed of "
                        "single-syllable training clips, so few speakers sound like many")
    p.add_argument("--praat-augment", action="store_true",
                   help="like --voice-augment, but shifts vowel resonances (x0.88-1.13) and pitch "
                        "(+-4 semitones) separately, with Praat; needs praat-parselmouth. "
                        "Use instead of --voice-augment, not with it")
    p.add_argument("--per-head-mix", action="store_true",
                   help="give each head its own mix of the encoder's layers instead of one shared mix")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="train only the layer mix and the heads; the encoder stays as it is")
    p.add_argument("--sentences", nargs="+", default=[],
                   help="sentence manifests (AISHELL) to mix in alongside --train, so the encoder "
                        "keeps hearing many speakers while it learns single syllables")
    p.add_argument("--sentence-hours", type=float, default=3.0,
                   help="hours of --sentences audio used per epoch; a new random selection each epoch")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.praat_augment and args.voice_augment:
        raise SystemExit("use --praat-augment or --voice-augment, not both")
    voice_mode = "praat" if args.praat_augment else args.voice_augment
    if voice_mode == "praat":
        import parselmouth  # noqa: F401  fail now, not inside a data-loader worker

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    if args.init_from:
        model = PinyinCTC.load(args.init_from, mask_time_prob=args.mask_time_prob,
                               per_head_mix=args.per_head_mix)
        feature_extractor = PinyinCTC.load_feature_extractor(args.init_from)
    else:
        model = PinyinCTC.from_encoder(args.encoder, mask_time_prob=args.mask_time_prob,
                                       per_head_mix=args.per_head_mix)
        feature_extractor = PinyinCTC.load_feature_extractor(args.encoder)
    model.freeze_front_end()   # the convolutional front end stays fixed
    if args.freeze_encoder:
        model.freeze_encoder()
    model.to(device)

    collator = Collator(feature_extractor)
    dev_loader = make_loader(read_manifests(args.dev), collator, args, shuffle=False)

    if args.eval_only:
        show("eval", evaluate(model, dev_loader, device, use_amp))
        return
    if not args.train:
        raise SystemExit("--train is required unless --eval-only is set")

    train_rows = read_manifests(args.train)
    sentence_rows = read_manifests(args.sentences) if args.sentences else []

    def epoch_loader(epoch):
        """Every epoch uses all of --train plus a fresh selection of sentences."""
        rows = train_rows + sample_hours(sentence_rows, args.sentence_hours, args.seed + epoch)
        return make_loader(rows, collator, args, shuffle=True, augment=args.augment,
                           voice_augment=voice_mode, seed=args.seed + epoch)

    train_loader = epoch_loader(1)
    hours = sum(r["duration"] for r in train_rows) / 3600
    print(f"{len(train_rows)} training clips, {hours:.1f} hours, "
          f"{len(train_loader)} batches per epoch, device={device}, amp={use_amp}, "
          f"voice augmentation={voice_mode or 'off'}", flush=True)
    if sentence_rows:
        print(f"plus {args.sentence_hours:.1f} hours per epoch drawn from "
              f"{len(sentence_rows)} sentence clips", flush=True)

    encoder_params = [q for q in model.encoder.parameters() if q.requires_grad]
    head_params = model.top_parameters()
    params = encoder_params + head_params
    groups = [{"params": head_params, "lr": args.head_lr}]
    if encoder_params:                       # empty when --freeze-encoder is set
        groups.insert(0, {"params": encoder_params, "lr": args.lr})
    print(f"training {sum(q.numel() for q in params) / 1e6:.2f}M parameters"
          + (" (encoder frozen)" if args.freeze_encoder else ""), flush=True)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)

    # Warm up, hold at full rate until halfway, then decay linearly to 5%.
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup = min(args.warmup_steps, max(1, total_steps // 10))
    hold_end = total_steps // 2

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        if step < hold_end:
            return 1.0
        return max(0.05, (total_steps - step) / max(1, total_steps - hold_end))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    weights = {"initial": 1.0, "final": 1.0, "tone": args.tone_weight}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    show("before training", evaluate(model, dev_loader, device, use_amp))

    best, step = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        if sentence_rows and epoch > 1:
            train_loader = epoch_loader(epoch)
        recent = {h: 0.0 for h in HEADS}   # loss over the last 50 batches only
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader, 1):
            logits, frame_lengths = run_model(model, batch, device, use_amp)
            labels = {h: batch["labels"][h].to(device) for h in HEADS}
            losses = ctc_losses(logits, frame_lengths, labels, batch["label_lengths"])
            loss = sum(weights[h] * losses[h] for h in HEADS) / args.grad_accum
            scaler.scale(loss).backward()
            for h in HEADS:
                recent[h] += losses[h].item()
            if i % args.grad_accum == 0 or i == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step += 1
            if i % 50 == 0:
                print(f"epoch {epoch} batch {i}/{len(train_loader)} step {step} "
                      + " ".join(f"{h}={recent[h] / 50:.3f}" for h in HEADS)
                      + f" lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
                recent = {h: 0.0 for h in HEADS}

        if epoch % args.eval_every and epoch != args.epochs:
            continue
        metrics = evaluate(model, dev_loader, device, use_amp)
        show(f"epoch {epoch}", metrics)
        if model.per_head_mix:   # which layers each head has come to rely on
            for h, mix in model.layer_mix().items():
                print(f"  {h:7} layer mix: " + " ".join(f"{w:.2f}" for w in mix), flush=True)
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps({"epoch": epoch, **metrics}) + "\n")
        model.save(out / "last", feature_extractor)
        if metrics["mean_error"] < best:
            best = metrics["mean_error"]
            model.save(out / "best", feature_extractor)
            print(f"  saved new best -> {out / 'best'}")


if __name__ == "__main__":
    main()