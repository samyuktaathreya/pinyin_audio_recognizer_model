# ============================================================== CELL 1: raw formant tracks (≈3 min, saved to Drive)
# Re-measures F1/F2/F3 in Hz at 10%, 20%, ... 90% of the voiced region, with the same settings as
# before, so the earlier features can be checked against raw values.
import json, math, os, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
from pinyin_units import split_syllable

runs = "/content/drive/MyDrive/pinyin_runs"
MANIFESTS = ["manifests/syl_train.jsonl", "manifests/good/train.jsonl",
             "manifests/syl_test.jsonl", "manifests/good/test_studycli.jsonl", "manifests/good/test_uvm.jsonl"]
FINALS = ("eng", "ang", "en", "an")
RAW_CSV = f"{runs}/formant_raw.csv"
FRACS = [i / 10 for i in range(1, 10)]

def raw_track(audio):
    import parselmouth
    from data import load_audio
    out = {"audio": audio}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            snd = parselmouth.Sound(load_audio(audio).astype(np.float64), sampling_frequency=16000)
            pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
            f0, times = pitch.selected_array["frequency"], pitch.xs()
            voiced = times[f0 > 0]
            if len(voiced) < 5:
                return out
            t0, t1 = float(voiced[0]), float(voiced[-1])
            med = float(np.median(f0[f0 > 0]))
            ceiling = 5500 if med > 165 else 5000
            fmt = snd.to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=ceiling)
            out.update(t0=t0, t1=t1, f0=med, ceiling=ceiling)
            for frac in FRACS:
                t = t0 + frac * (t1 - t0)
                for i in (1, 2, 3):
                    out[f"F{i}_{int(frac * 100)}"] = np.nanmedian(
                        [fmt.get_value_at_time(i, t + d) for d in (-0.01, 0, 0.01)])
    except Exception:
        pass
    return out

rows = []
for m in MANIFESTS:
    for line in open(m):
        r = json.loads(line)
        p = split_syllable(r["pinyin"])
        if p and p[1] in FINALS:
            rows.append({"audio": r["audio"], "voice": r.get("speaker") or r.get("source"),
                         "pinyin": r["pinyin"], "final": p[1]})
clips = pd.DataFrame(rows).drop_duplicates("audio").set_index("audio")

if os.path.exists(RAW_CSV):
    raw = pd.read_csv(RAW_CSV).set_index("audio")
else:
    with Pool(2) as pool:
        raw = pd.DataFrame(pool.map(raw_track, list(clips.index), chunksize=16)).set_index("audio")
    raw.to_csv(RAW_CSV)
d = clips.join(raw)

# what 6b + 7 heard (held-out voices only), to mark the misheard clips
Ps = []
for name in ["stage6b", "stage7_hubert"]:
    x = pd.read_csv(f"{runs}/{name}_probs.csv", dtype=str, keep_default_na=False).set_index("audio")
    cols = [c for c in x.columns if c.startswith("final:") and "silence" not in c]
    Ps.append(np.exp(x[cols].astype(float)))
common = Ps[0].index.intersection(Ps[1].index)
d["heard"] = ((Ps[0].loc[common] + Ps[1].loc[common]) / 2).idxmax(axis=1).str.split(":").str[1].reindex(d.index)
d["misheard"] = d.heard.notna() & (d.heard != d.final)

# consistency with the features the rescorer used: F1/F3 at 50% should equal its r13_50
old = pd.read_csv(f"{runs}/formant_feats.csv").set_index("audio")["r13_50"]
diff = (d.F1_50 / d.F3_50 - old.reindex(d.index)).abs()
print(f"{len(d)} clips. Re-measured F1/F3 matches the saved r13_50 on {(diff < 1e-6).mean():.0%} "
      f"(should be ~100%: same code, same settings)")
print(d.groupby("voice")[["f0", "ceiling"]].median().round(0).to_string())


# ============================================================== CELL 2: F1 vs F2 per voice + within-speaker test
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

PAIR = ("eng", "ang")      # or ("en", "an")
POINT = 50                 # % of the voiced region to plot (try 30 too: the nasal is in the voiced region)
COL = {PAIR[0]: "#2a78d6", PAIR[1]: "#eb6834"}

x = d[d.final.isin(PAIR)]
voices = sorted(x.voice.unique(), key=lambda v: (v in ("FV3", "MV3", "studycli", "uvm"), v))
nc = 4
fig, axes = plt.subplots(math.ceil(len(voices) / nc), nc, figsize=(16, 4 * math.ceil(len(voices) / nc)))
for ax, v in zip(axes.flat, voices):
    y = x[x.voice == v]
    for fin in PAIR:
        z = y[(y.final == fin) & ~y.misheard]
        ax.scatter(z[f"F2_{POINT}"], z[f"F1_{POINT}"], s=22, color=COL[fin], alpha=0.7, label=fin)
    z = y[y.misheard & (y.final == PAIR[0])]
    if len(z):
        ax.scatter(z[f"F2_{POINT}"], z[f"F1_{POINT}"], s=60, marker="x", color="#111",
                   label=f"{PAIR[0]} heard as {PAIR[1]}")
    ax.invert_xaxis(); ax.invert_yaxis()          # phonetics convention: open vowels low, front vowels left
    ax.set_title(f"{v}  (f0 {y.f0.median():.0f} Hz, ceiling {y.ceiling.median():.0f})", fontsize=10)
    ax.set_xlabel("F2 (Hz)"); ax.set_ylabel("F1 (Hz)")
    ax.grid(color="#ddd", lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
for ax in axes.flat[len(voices):]:
    ax.axis("off")
h, l = [], []
for ax in axes.flat[:len(voices)]:
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in l:
            h.append(hh); l.append(ll)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.suptitle(f"F1 vs F2 at {POINT}% of the voiced region", y=0.995)
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=3, frameon=False)
plt.show()

# within-speaker: train and test on one voice's own clips (5-fold cross-validation)
FEATS = [f"F{i}_{p}" for i in (1, 2, 3) for p in (30, 50, 70)]
clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
rows = []
for v in voices:
    y = x[x.voice == v]
    lab = (y.final == PAIR[0]).values
    if min(lab.sum(), (~lab).sum()) < 5:
        continue
    pred = cross_val_predict(clf, y[FEATS], lab, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    mis = (y.misheard & (y.final == PAIR[0])).values
    rows.append({"voice": v, "clips": len(y), "within-voice accuracy": f"{(pred == lab).mean():.0%}",
                 f"misheard {PAIR[0]} put right": f"{(pred[mis] == lab[mis]).sum()}/{mis.sum()}" if mis.any() else "-"})
print(f"Formants only, {PAIR[0]} vs {PAIR[1]}, trained and tested within each voice (chance = 50%)")
print(pd.DataFrame(rows).to_string(index=False))


# ============================================================== CELL 3: look at clips by hand
# Spectrogram with Praat's formant tracks, the voiced region, the 20/50/80% points, and the numbers
# the script saved. Plays each clip. Pick clips with the lists at the bottom.
from IPython.display import Audio, display
import parselmouth
from data import load_audio

def inspect(audio):
    r = d.loc[audio]
    snd = parselmouth.Sound(load_audio(audio).astype(np.float64), sampling_frequency=16000)
    spec = snd.to_spectrogram(window_length=0.005, maximum_frequency=5000)
    fmt = snd.to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=r.ceiling)
    db = 10 * np.log10(np.maximum(spec.values, 1e-12))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.pcolormesh(spec.xs(), spec.ys(), db, cmap="Greys", vmin=db.max() - 70, shading="auto")
    ts = [t for t in fmt.xs() if r.t0 - 0.05 <= t <= r.t1 + 0.05]   # tracks in silence are meaningless
    for i, c in zip((1, 2, 3), ("#e34948", "#1baf7a", "#2a78d6")):
        ax.scatter(ts, [fmt.get_value_at_time(i, t) for t in ts], s=4, color=c, label=f"F{i} track")
    ax.axvspan(r.t0, r.t1, color="#eda100", alpha=0.08, label="voiced region")
    for p in (20, 50, 80):
        t = r.t0 + p / 100 * (r.t1 - r.t0)
        ax.axvline(t, color="#eda100", lw=1)
        ax.text(t, 4850, f"{p}%", ha="center", fontsize=8)
    ax.set_ylim(0, 5000); ax.set_xlabel("time (s)"); ax.set_ylabel("Hz")
    heard = f", model heard {r.heard}" if isinstance(r.heard, str) else ""
    ax.set_title(f"{r.voice} {r.pinyin}{heard}", fontsize=10)
    ax.legend(loc="upper right", fontsize=7, ncol=5, frameon=False)
    plt.show()
    print("  saved:  " + "   ".join(f"{p}%: F1 {r[f'F1_{p}']:.0f}  F2 {r[f'F2_{p}']:.0f}  F3 {r[f'F3_{p}']:.0f}"
                                     for p in (20, 50, 80)))
    display(Audio(load_audio(audio), rate=16000))

VOICE = "studycli"
mis = d[(d.voice == VOICE) & (d.final == "eng") & d.misheard].index[:4]
ok = d[(d.voice == VOICE) & (d.final == "ang") & ~d.misheard].index[:2]
for a in list(mis) + list(ok):
    inspect(a)