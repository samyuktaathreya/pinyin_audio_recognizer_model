"""Pitch-contour vote on tone 2 vs tone 3, used by score.py to re-split the model's 2-vs-3 probability.

The contour is measured one clip at a time, in semitones relative to the start of the syllable, so it
needs no calibration. A small logistic regression, trained on the training voices only, turns six
shape measures into log-odds for tone 2 over tone 3. Same code as notebook cell 10, so the saved
classifier sees exactly the features it was trained on.

Train and save (notebook):  see the "save the pitch classifier" cell.
Use:                        vote = PitchTone(path).vote(wav)   # None if no pitch was found
"""
import math
import warnings

import numpy as np

N = 20                                   # contour resampled to 20 points over the voiced part
SHAPE = ["dip", "dip_pos", "rise", "net", "first_half", "creak"]
SAMPLE_RATE = 16000


def contour(wav, sr=SAMPLE_RATE):
    """(20-point contour in semitones from the syllable start, creak fraction), or None."""
    import parselmouth
    from scipy.signal import medfilt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snd = parselmouth.Sound(np.asarray(wav, dtype=np.float64), sampling_frequency=sr)
        f = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600).selected_array["frequency"]
        if (f > 0).sum() < 8:
            return None
        med = float(np.median(f[f > 0]))
        # second pass in a range around this clip's own pitch: avoids most octave jumps
        p = snd.to_pitch(time_step=0.01, pitch_floor=max(60, med / 2.2), pitch_ceiling=min(800, med * 2.2))
        f = p.selected_array["frequency"]
        v = np.where(f > 0)[0]
        if len(v) < 8:
            return None
        a, b = v[0], v[-1]
        seg = f[a:b + 1].astype(float)
        gaps = seg <= 0                                # unvoiced frames inside the syllable (creak)
        st = np.where(gaps, np.nan, 12 * np.log2(np.maximum(seg, 1) / med))
        idx = np.arange(len(st))
        st = np.interp(idx, idx[~gaps], st[~gaps])
        sm = medfilt(st, 7)
        bad = np.abs(st - sm) > 4                      # leftover jumps: tracking errors
        st = np.interp(idx, idx[~bad], st[~bad]) if (~bad).sum() >= 4 else sm
        st = medfilt(st, 5)
        c = np.interp(np.linspace(0, len(st) - 1, N), idx, st)
        c = c - c[1:4].mean()
        mid = gaps[int(0.3 * len(gaps)):int(0.7 * len(gaps)) + 1]
        return c, (float(mid.mean()) if len(mid) else 0.0)


def features_from_contour(c, creak):
    c = np.asarray(c, dtype=float)
    inner = c[2:-2]
    end = c[-3:].mean()
    return {"dip": -inner.min(), "dip_pos": (inner.argmin() + 2) / (N - 1), "rise": end - inner.min(),
            "net": end, "first_half": c[N // 2], "creak": creak}


def make_classifier():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.5, max_iter=3000))


class PitchTone:
    def __init__(self, path):
        import joblib
        saved = joblib.load(path)
        self.clf, self.features = saved["clf"], saved["features"]

    def vote(self, wav):
        """log-odds of tone 2 over tone 3 from the pitch contour, or None if no pitch was found."""
        try:
            got = contour(wav)
        except Exception:
            return None
        if got is None:
            return None
        f = features_from_contour(*got)
        p2 = float(self.clf.predict_proba(np.array([[f[k] for k in self.features]]))[0, 1])
        return math.log(p2 + 1e-6) - math.log(1 - p2 + 1e-6)