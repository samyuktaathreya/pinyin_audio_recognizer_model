"""One shared wav2vec2 encoder with three CTC heads: initial, final, tone.

The heads read a learned weighted mix of ALL the encoder's layers, not just the last one.
In a pretrained-only checkpoint the top layers are specialised for the pretraining task and
carry little information about which sound is being spoken, so reading only the last layer
leaves the heads almost nothing to learn from.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from pinyin_units import HEADS, VOCABS

DEFAULT_ENCODER = "TencentGameMate/chinese-wav2vec2-base"


class PinyinCTC(nn.Module):
    def __init__(self, encoder, dropout=0.1, per_head_mix=False):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        n_layers = encoder.config.num_hidden_layers + 1          # + the input to layer 1
        # One mix shared by the three heads (shape: layers), or one mix for each head
        # (shape: heads x layers) so that, say, the tone head can lean on different
        # layers than the initial head. Zeros = an equal mix.
        self.per_head_mix = per_head_mix
        shape = (len(HEADS), n_layers) if per_head_mix else (n_layers,)
        self.layer_weights = nn.Parameter(torch.zeros(shape))
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {h: nn.Linear(hidden, len(VOCABS[h])) for h in HEADS})

    @classmethod
    def from_encoder(cls, name=DEFAULT_ENCODER, mask_time_prob=0.05, dropout=0.1,
                     per_head_mix=False):
        """Fresh model: pretrained encoder + untrained layer mix and heads."""
        encoder = Wav2Vec2Model.from_pretrained(
            name,
            apply_spec_augment=True,
            mask_time_prob=mask_time_prob,   # the checkpoint's own value is for pretraining
            mask_time_length=10,
            mask_feature_prob=0.0,
            layerdrop=0.0,
        )
        return cls(encoder, dropout, per_head_mix)

    @classmethod
    def load(cls, model_dir, mask_time_prob=None, per_head_mix=False):
        """Load a model written by save().

        A model saved with one mix per head always loads that way. per_head_mix=True also
        converts a model saved with a single shared mix (such as stage 2): every head
        starts from a copy of the shared mix, so the model's answers are unchanged until
        training moves the copies apart."""
        model_dir = Path(model_dir)
        kwargs = {} if mask_time_prob is None else {"mask_time_prob": mask_time_prob}
        encoder = Wav2Vec2Model.from_pretrained(model_dir / "encoder", **kwargs)
        top = model_dir / "top.pt"
        if not top.exists():
            raise FileNotFoundError(f"{top} not found; this model was saved by an older version "
                                    "of the code and can't be loaded. Retrain it.")
        state = torch.load(top, map_location="cpu")
        saved_per_head = state["layer_weights"].dim() == 2
        if per_head_mix and not saved_per_head:
            state["layer_weights"] = state["layer_weights"].repeat(len(HEADS), 1)
        model = cls(encoder, per_head_mix=per_head_mix or saved_per_head)
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("encoder.")]
        if missing or unexpected:
            raise RuntimeError(f"bad checkpoint: missing={missing} unexpected={unexpected}")
        return model

    def save(self, model_dir, feature_extractor=None):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(model_dir / "encoder")
        top = {k: v for k, v in self.state_dict().items() if not k.startswith("encoder.")}
        torch.save(top, model_dir / "top.pt")
        (model_dir / "units.json").write_text(json.dumps(VOCABS, indent=1))
        if feature_extractor is not None:
            feature_extractor.save_pretrained(model_dir / "encoder")

    def top_parameters(self):
        """Everything trained from scratch: layer mix, norm and heads."""
        return [p for n, p in self.named_parameters() if not n.startswith("encoder.")]

    def layer_mix(self):
        """Share of each layer in the mix: a list, or {head: list} with one mix per head."""
        mix = F.softmax(self.layer_weights, dim=-1).tolist()
        return dict(zip(HEADS, mix)) if self.per_head_mix else mix

    @staticmethod
    def load_feature_extractor(name_or_dir):
        path = Path(name_or_dir) / "encoder"
        return Wav2Vec2FeatureExtractor.from_pretrained(path if path.exists() else name_or_dir)

    def frame_lengths(self, sample_lengths):
        """Number of output frames (about one per 20 ms) for each raw audio length."""
        return self.encoder._get_feat_extract_output_lengths(sample_lengths)

    def forward(self, input_values, attention_mask=None):
        out = self.encoder(input_values, attention_mask=attention_mask,
                           output_hidden_states=True)
        layers = torch.stack(out.hidden_states, dim=0)                # (L, B, T, D)
        weights = F.softmax(self.layer_weights, dim=-1).to(layers.dtype)
        if not self.per_head_mix:
            hidden = (weights.view(-1, 1, 1, 1) * layers).sum(dim=0)
            hidden = self.dropout(self.norm(hidden))
            return {h: head(hidden) for h, head in self.heads.items()}   # each (B, T, vocab)
        logits = {}
        for i, h in enumerate(HEADS):
            hidden = (weights[i].view(-1, 1, 1, 1) * layers).sum(dim=0)
            logits[h] = self.heads[h](self.dropout(self.norm(hidden)))
        return logits


def ctc_losses(logits, frame_lengths, labels, label_lengths):
    """One CTC loss per head. Sum them for the training loss."""
    losses = {}
    for h in HEADS:
        log_probs = F.log_softmax(logits[h].float(), dim=-1).transpose(0, 1)   # (T, B, C)
        losses[h] = F.ctc_loss(log_probs, labels[h], frame_lengths, label_lengths,
                               blank=0, zero_infinity=True)
    return losses


def greedy_decode(logits, frame_lengths):
    """Best-path CTC decoding. Returns {head: [list of ids per utterance]}."""
    out = {}
    for h in HEADS:
        best = logits[h].argmax(dim=-1).cpu()
        sequences = []
        for row, n in zip(best, frame_lengths.tolist()):
            ids = torch.unique_consecutive(row[:n]).tolist()
            sequences.append([i for i in ids if i != 0])
        out[h] = sequences
    return out