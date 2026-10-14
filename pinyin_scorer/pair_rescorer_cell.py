import json, math, os, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pinyin_units import split_syllable

runs = "/content/drive/MyDrive/pinyin_runs"
PROB_CSVS = [f"{runs}/stage6b_probs.csv", f"{runs}/stage7_hubert_probs.csv"]   # averaged = the 6b+7 model
TRAIN_MANIFESTS = ["manifests/syl_train.jsonl", "manifests/good/train.jsonl"]
TEST_VOICES = ["FV3", "MV3", "studycli", "uvm"]
PAIRS = [("eng", "ang"), ("en", "an")]
PAIR_FINALS = {f for p in PAIRS for f in p}
THR = 0.05
FEATS_CSV = f"{runs}/formant_feats.csv"          # cached, so reruns skip the extraction

# ------------------------------------------------------------------ 1. formant features per clip
def bark(f):
    return 26.81 * f / (1960 + f) - 0.53          # Traunmüller

def clip_features(audio):
    """Speaker-relative formant measures at 20/50/80% of the voiced region."""
    import parselmouth
    from data import load_audio
    out = {"audio": audio, "ok": 0}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            snd = parselmouth.Sound(load_audio(audio).astype(np.float64), sampling_frequency=16000)
            pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
            f0, times = pitch.selected_array["frequency"], pitch.xs()
            voiced = times[f0 > 0]
            if len(voiced) < 5:
                return out
            t0, t1 = voiced[0], voiced[-1]
            dur, med = t1 - t0, float(np.median(f0[f0 > 0]))
            fmt = snd.to_formant_burg(time_step=0.005, max_number_of_formants=5,
                                      maximum_formant=5500 if med > 165 else 5000)
            out.update(ok=1, log_dur=math.log(dur), bad_points=0)
            for tag, frac in [("20", 0.2), ("50", 0.5), ("80", 0.8)]:
                t = t0 + frac * dur
                F1, F2, F3 = [np.nanmedian([fmt.get_value_at_time(i, t + d) for d in (-0.01, 0, 0.01)])
                              for i in (1, 2, 3)]
                if not (np.isfinite([F1, F2, F3]).all() and F1 < F2 < F3):
                    out["bad_points"] += 1
                    F1 = F2 = F3 = np.nan
                out[f"b21_{tag}"] = bark(F2) - bark(F1)      # F2 - F1 in Bark
                out[f"b32_{tag}"] = bark(F3) - bark(F2)      # F3 - F2 in Bark
                out[f"r13_{tag}"] = F1 / F3                  # F3 as the speaker-size ruler
                out[f"r23_{tag}"] = F2 / F3
            bw = fmt.get_bandwidth_at_time(1, t0 + 0.5 * dur)
            out["log_bw1"] = math.log(bw) if bw and np.isfinite(bw) and bw > 0 else np.nan
    except Exception:
        out["ok"] = 0
    return out

# model probabilities (6b + 7 averaged) for the four held-out voices
Ps, meta = [], None
for path in PROB_CSVS:
    d = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("audio")
    cols = [c for c in d.columns if c.startswith("final:")]
    Ps.append(np.exp(d[cols].astype(float)))
    meta = d[["source", "target", "target_final", "target_tone"]] if meta is None else meta
common = Ps[0].index
for p in Ps[1:]:
    common = common.intersection(p.index)
P = sum(p.loc[common] for p in Ps) / len(Ps)
meta = meta.loc[common]
test = meta[meta.target_final.isin(PAIR_FINALS) & meta.source.isin(TEST_VOICES)].copy()

# training voices: only their audio is used (no model probabilities, those models trained on them)
rows = []
for m in TRAIN_MANIFESTS:
    for line in open(m):
        r = json.loads(line)
        parts = split_syllable(r["pinyin"])
        if parts and parts[1] in PAIR_FINALS:
            rows.append({"audio": r["audio"], "source": r.get("speaker") or r.get("source"),
                         "target": r["pinyin"], "target_final": parts[1], "target_tone": str(parts[2])})
train = pd.DataFrame(rows).set_index("audio")

paths = sorted(set(test.index) | set(train.index))
feats = pd.read_csv(FEATS_CSV).set_index("audio") if os.path.exists(FEATS_CSV) else pd.DataFrame()
todo = [a for a in paths if a not in feats.index]
print(f"{len(test)} held-out clips, {len(train)} training-voice clips; extracting formants for {len(todo)}")
if todo:
    with Pool(2) as pool:
        new = pd.DataFrame(pool.map(clip_features, todo, chunksize=16)).set_index("audio")
    feats = pd.concat([feats, new])
    feats.to_csv(FEATS_CSV)
print(f"formants tracked on {feats.loc[paths].ok.mean():.0%} of clips; "
      f"all three points valid on {(feats.loc[paths].bad_points == 0).mean():.0%}")

# duration relative to the usual length of that tone (tone 3 is much longer than tone 4)
tone_mean = feats.loc[train.index].join(train.target_tone).groupby("target_tone").log_dur.mean()
for df in (test, train):
    df[feats.columns] = feats.loc[df.index]
    df["dur_rel"] = df.log_dur - df.target_tone.map(tone_mean)

FORM = [f"{k}_{t}" for k in ("b21", "b32", "r13", "r23") for t in ("20", "50", "80")] \
       + ["log_bw1", "bad_points", "dur_rel"]

def classifier():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))

def model_logit(idx, a, b):
    return np.log(P.loc[idx, f"final:{a}"] + 1e-6) - np.log(P.loc[idx, f"final:{b}"] + 1e-6)

# ------------------------------------------------------------------ 2. formants alone (diagnostic)
print("\n=== Formant features ALONE, trained on the 8 training voices, tested on the 4 held-out voices")
print("    (does the information exist in a speaker-relative form?)")
for a, b in PAIRS:
    tr = train[train.target_final.isin([a, b]) & (train.ok == 1)]
    clf = classifier().fit(tr[FORM], tr.target_final == a)
    for v in TEST_VOICES:
        te = test[(test.source == v) & test.target_final.isin([a, b])]
        pred_a = clf.predict(te[FORM])
        right = pred_a == (te.target_final == a)
        model_wrong = (model_logit(te.index, a, b) > 0) != (te.target_final == a)
        print(f"  {a}/{b} {v:9} accuracy {right.mean():5.1%}   "
              f"on the {model_wrong.sum():2d} clips the model gets wrong: {right[model_wrong.values].sum()} right")

# ------------------------------------------------------------------ 3. the rescorer
# Features + the model's own log(p_a / p_b). Trained leave-one-voice-out on the held-out voices,
# because the model's probabilities on its own training voices are overconfident.
def rescore(fit_idx, apply_idx):
    """New final probabilities for apply_idx: the pair's total probability is kept, only its split
    between the two partners is re-decided. Every other unit is unchanged."""
    P2 = P.loc[apply_idx].copy()
    for a, b in PAIRS:
        tr = test.loc[fit_idx]
        tr = tr[tr.target_final.isin([a, b])]
        X = lambda idx: test.loc[idx, FORM].assign(model=model_logit(idx, a, b))
        clf = classifier().fit(X(tr.index), tr.target_final == a)
        q = clf.predict_proba(X(apply_idx))[:, 1]
        mass = P2[f"final:{a}"] + P2[f"final:{b}"]
        P2[f"final:{a}"], P2[f"final:{b}"] = mass * q, mass * (1 - q)
    return P2

def outcomes(Pm, idx):
    units = [c for c in Pm.columns if "silence" not in c]
    heard = Pm.loc[idx, units].idxmax(axis=1).str.split(":").str[1]
    tgt = test.loc[idx, "target_final"]
    p_t = np.array([Pm.at[i, f"final:{t}"] for i, t in zip(idx, tgt)])
    partner = tgt.map({a: b for a, b in PAIRS} | {b: a for a, b in PAIRS})
    p_p = np.array([Pm.at[i, f"final:{t}"] for i, t in zip(idx, partner)])
    return pd.DataFrame({"final": tgt.values, "heard": heard.values, "correct": (heard == tgt).values,
                         "flagged": p_t < THR, "swap_miss": p_p >= THR}, index=idx)

rows, fixes = [], []
for v in TEST_VOICES:
    held = test.index[test.source == v]
    before = outcomes(P, held)
    after = outcomes(rescore(test.index[test.source != v], held), held)
    for a, b in PAIRS:
        for x, y in [(a, b), (b, a)]:
            m = before.final == x
            rows.append({"voice": v, "clips said": x, "n": m.sum(),
                         "heard as partner before": (before.heard[m] == y).sum(),
                         "after": (after.heard[m] == y).sum(),
                         "fixed": (~before.correct[m] & after.correct[m]).sum(),
                         "broken": (before.correct[m] & ~after.correct[m]).sum(),
                         "false alarms before": before.flagged[m].sum(), "false alarms after": after.flagged[m].sum(),
                         "partner accepted before": before.swap_miss[m].sum(),
                         "partner accepted after": after.swap_miss[m].sum()})
res = pd.DataFrame(rows)
tot = res.drop(columns=["voice", "clips said"]).sum()
print("\n=== Rescorer (formants + model), each voice scored by a rescorer trained on the other three")
print("    'partner accepted' = this clip would pass as an attempt at the partner (missed swap): lower is better")
print(res.to_string(index=False))
print("\nTOTAL over all four voices:")
print(tot.to_string())

# ------------------------------------------------------------------ 4. save one trained on all four voices
import joblib
final = {}
for a, b in PAIRS:
    tr = test[test.target_final.isin([a, b])]
    X = tr[FORM].assign(model=model_logit(tr.index, a, b))
    final[(a, b)] = classifier().fit(X, tr.target_final == a)
joblib.dump({"pairs": final, "features": FORM + ["model"], "tone_mean": tone_mean.to_dict()},
            f"{runs}/pair_rescorer.joblib")
print(f"\nsaved {runs}/pair_rescorer.joblib (not used by score.py yet)")