"""Pinyin syllable -> (initial, final, tone) units, plus the three vocabularies.

Finals are stored in a canonical form so that the same sound always gets the
same label regardless of spelling:
  - zero-initial spellings are undone:  yi -> i, wu -> u, yu -> v, you -> iu, wei -> ui
  - u-umlaut is always "v":             ju -> j + v, lüe -> l + ve, yuan -> van
  - the two "buzzed" vowels are split from the real i:
        zi/ci/si       -> "ii"
        zhi/chi/shi/ri -> "iii"
"""
import re

BLANK = "<blank>"      # CTC blank, always id 0
NO_INITIAL = "_"       # syllables like "an", "er", "wo", "yi"

INITIALS = [NO_INITIAL,
            "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
            "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s"]

FINALS = ["a", "o", "e", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "ong", "er",
          "i", "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing", "iong", "io",
          "u", "ua", "uo", "uai", "ui", "uan", "un", "uang", "ueng",
          "v", "ve", "van", "vn",
          "ii", "iii"]

TONES = ["1", "2", "3", "4", "5"]   # 5 = neutral

HEADS = ("initial", "final", "tone")
VOCABS = {
    "initial": [BLANK] + INITIALS,
    "final": [BLANK] + FINALS,
    "tone": [BLANK] + TONES,
}
_TO_ID = {h: {u: i for i, u in enumerate(v)} for h, v in VOCABS.items()}

_SYLLABLE = re.compile(r"^([a-z]+)([1-5]?)$")
_TWO_LETTER = ("zh", "ch", "sh")
_ONE_LETTER = set("bpmfdtnlgkhjqxrzcs")
_LONG_FORMS = {"iou": "iu", "uei": "ui", "uen": "un"}
_FINAL_SET = set(FINALS)


def split_syllable(syllable):
    """'zhuang4' -> ('zh', 'uang', '4').  Returns None if it isn't a normal syllable."""
    s = syllable.strip().lower().replace("u:", "v").replace("ü", "v")
    m = _SYLLABLE.match(s)
    if not m:
        return None
    body, tone = m.group(1), m.group(2) or "5"

    if body.startswith("yu"):
        initial, final = NO_INITIAL, "v" + body[2:]
    elif body.startswith("yi"):
        initial, final = NO_INITIAL, "i" + body[2:]
    elif body.startswith("y"):
        initial, final = NO_INITIAL, "i" + body[1:]
    elif body.startswith("wu"):
        initial, final = NO_INITIAL, "u" + body[2:]
    elif body.startswith("w"):
        initial, final = NO_INITIAL, "u" + body[1:]
    elif body[:2] in _TWO_LETTER:
        initial, final = body[:2], body[2:]
    elif body[0] in _ONE_LETTER:
        initial, final = body[0], body[1:]
    else:
        initial, final = NO_INITIAL, body

    final = _LONG_FORMS.get(final, final)
    if initial in ("j", "q", "x") and final.startswith("u"):
        final = "v" + final[1:]
    if initial in ("n", "l") and final == "ue":
        final = "ve"
    if final == "i":
        if initial in ("z", "c", "s"):
            final = "ii"
        elif initial in ("zh", "ch", "sh", "r"):
            final = "iii"

    if final not in _FINAL_SET:
        return None
    return initial, final, tone


def encode(pinyin):
    """'guang3 zhou1' -> {'initial': [...], 'final': [...], 'tone': [...]} of ids, or None."""
    out = {h: [] for h in HEADS}
    for syllable in pinyin.split():
        parts = split_syllable(syllable)
        if parts is None:
            return None
        for head, unit in zip(HEADS, parts):
            out[head].append(_TO_ID[head][unit])
    return out if out["tone"] else None


def display(head, unit):
    """Unit -> something to show a learner."""
    if head == "initial" and unit == NO_INITIAL:
        return "(none)"
    if head == "final":
        if unit in ("ii", "iii"):
            return "i"
        return unit.replace("v", "ü")
    if head == "tone" and unit == "5":
        return "neutral"
    return unit


_HANZI_ONLY = re.compile(r"^[一-鿿]+$")


_NUMERALS = set("一二三四五六七八九十百千万亿零两")


def apply_sandhi(chars, syllables):
    """Change dictionary tones to spoken tones.

    pypinyin's own tone_sandhi option misses common cases (it leaves 产品 as chan3 pin3),
    so the three standard rules are applied here:
      - 不 is bu2 before a 4th tone, otherwise bu4
      - 一 is yi2 before a 4th tone and yi4 before other tones, but stays yi1 when it is
        part of a number, an ordinal (第一) or the last syllable
      - in a run of 3rd tones, every one except the last becomes a 2nd tone
    """
    tones = [s[-1] for s in syllables]
    bases = [s[:-1] for s in syllables]
    n = len(chars)
    for i, ch in enumerate(chars):
        nxt = tones[i + 1] if i + 1 < n else None
        if ch == "不" and bases[i] == "bu":
            tones[i] = "2" if nxt == "4" else "4"
        elif ch == "一" and bases[i] == "yi":
            is_number = (nxt is None
                         or (i > 0 and (chars[i - 1] in _NUMERALS or chars[i - 1] == "第"))
                         or chars[i + 1] in _NUMERALS)
            if is_number:
                tones[i] = "1"
            elif nxt in ("4", "5"):
                tones[i] = "2"
            else:
                tones[i] = "4"
    i = 0
    while i < n:
        if tones[i] == "3":
            j = i
            while j + 1 < n and tones[j + 1] == "3":
                j += 1
            for k in range(i, j):
                tones[k] = "2"
            i = j + 1
        else:
            i += 1
    return [b + t for b, t in zip(bases, tones)]


def hanzi_to_pinyin(text, sandhi=True):
    """'广州 房产' -> 'guang3 zhou1 fang2 chan3', or None if anything is unusual.

    With sandhi=True the labels follow what is actually spoken
    (3+3 -> 2+3, and the tone changes of yi and bu), not the dictionary tone.
    """
    from pypinyin import Style, lazy_pinyin

    chars = re.sub(r"\s+", "", text)
    if not chars or not _HANZI_ONLY.match(chars):
        return None
    syllables = lazy_pinyin(chars, style=Style.TONE3, neutral_tone_with_five=True)
    if len(syllables) != len(chars):
        return None
    if any(not s or s[-1] not in "12345" for s in syllables):
        return None
    if sandhi:
        syllables = apply_sandhi(chars, syllables)
    if any(split_syllable(s) is None for s in syllables):
        return None
    return " ".join(syllables)
