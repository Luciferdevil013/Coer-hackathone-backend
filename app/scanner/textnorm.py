"""Free-text normalisation shared by the scanner (near-copy detection) and the dedupe operator.

Two levels, both deterministic:
  * basic     - HTML codes decoded, case and spacing ignored ("Same  text" == "same text")
  * content   - also drops links, retweet prefixes ("RT @user:"), @/# signs, punctuation and
                trailing ellipses, so a retweet or a re-shared link compares equal to the original
"""
from __future__ import annotations

import html
import re
import unicodedata

HTML_ENTITY = re.compile(r"&(?:[A-Za-z][A-Za-z0-9]{1,9}|#\d{2,7}|#[xX][0-9A-Fa-f]{2,6});")
_URL = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|co|in|ly|io|me)/\S*", re.I)
_RT = re.compile(r"^(?:\s*(?:RT|MT)\s*:?\s*@?\w+\s*:\s*)+", re.I)
_QUOTE_MENTION = re.compile(r"^[\"“”']*@\w+\s*:\s*")
_ELLIPSIS = re.compile(r"(?:…|\.\.\.)\s*$")
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
MIN_CONTENT_CHARS = 15   # shorter than this ("lol", "thanks!") is not distinctive enough to call a copy


def has_html_entities(v: str) -> bool:
    return bool(HTML_ENTITY.search(v)) and html.unescape(v) != v


def basic(v) -> str:
    if v is None:
        return ""
    s = unicodedata.normalize("NFKC", html.unescape(str(v)))
    return re.sub(r"\s+", " ", s).strip().lower()


def content(v) -> str:
    s = basic(v)
    if not s:
        return ""
    s = _URL.sub(" ", s)
    s = _RT.sub("", s)
    s = _QUOTE_MENTION.sub("", s)
    s = _ELLIPSIS.sub("", s.strip())
    s = _NON_WORD.sub(" ", s).replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) >= MIN_CONTENT_CHARS else ""


# ---------------------------------------------------------------- encoding repair (mojibake, lost characters)

# a UTF-8 lead byte read as Latin-1/cp1252 ("Ã©" for "é", "â€™" for "’", "Â " for a no-break space), C1 control codes,
# and the replacement character all mean the text was decoded with the wrong encoding
_BAD = re.compile(r"[\u0080-\u009f]|�|[ÃÂ][\u0080-¿]|â€|[ÐÑÓ][\u0080-¿]")
# what is left once repair fails: a run of Latin-1 high characters that contains one of the markers above
_JUNK = re.compile(r"[\u0080-ÿ�]{2,}")
_LOST_FIELD = re.compile(r"\?{2,}")                  # '??', '??????' = characters lost when the file was saved
_LOST_PROSE = re.compile(r"(?<!\S)\?{4,}(?!\S)")     # in prose only a standalone run: "really???" is punctuation
_MIXED_SCRIPT = re.compile(r"[A-Za-z][Ѐ-ӿ]|[Ѐ-ӿ][A-Za-z]")
_CYRILLIC_MOJIBAKE = re.compile(r"[\u0400-\u04FF\?]{2,}")


def _badness(s: str) -> int:
    return len(_BAD.findall(s))


def _undo_misdecoding(s: str) -> str:
    """Re-decode text that was UTF-8 read as cp1252/Latin-1, as many layers deep as it helps.
    A repair is kept only if it removes markers without inventing a mixed-script jumble."""
    for _ in range(3):
        better = None
        for enc in ("cp1252", "latin-1"):
            try:
                t = s.encode(enc).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if _badness(t) < _badness(s) and not _MIXED_SCRIPT.search(t):
                better = t
                break
        if better is None:
            return s
        s = better
    return s


def repair_text(v: str, prose: bool = False) -> str:
    """Fix garbled characters: repair what can be repaired, remove what cannot. Returns `v` unchanged when clean."""
    s = str(v)
    has_cyrillic = bool(not prose and _CYRILLIC_MOJIBAKE.search(s))
    if not (_BAD.search(s) or "?" in s or has_cyrillic):
        return s
    t = _undo_misdecoding(s) if _BAD.search(s) else s
    t = _JUNK.sub(lambda m: " " if _BAD.search(m.group(0)) else m.group(0), t)
    t = t.replace("\ufffd", " ")
    if not prose:
        t = re.sub(r"\bLakm\?{1,2}(?=\s|$)", "Lakmé", t, flags=re.I)
        t = _CYRILLIC_MOJIBAKE.sub(" ", t)
        t = _LOST_FIELD.sub(" ", t)
    else:
        t = _LOST_PROSE.sub(" ", t)
    t = re.sub(r"[\u0080-\u009f]", "", t).replace(" ", " ")
    if t == s:
        return s
    # tidy only around what was removed: "Cooler??????(White" -> "Cooler (White", never touch other spacing
    t = re.sub(r" {2,}", " ", t)
    t = re.sub(r" ([,.;:!?)\]])", r"\1", t)
    return t.strip()


def is_garbled(v: str, prose: bool = False) -> bool:
    s = str(v)
    return bool(_BAD.search(s) or (_LOST_PROSE if prose else _LOST_FIELD).search(s))
