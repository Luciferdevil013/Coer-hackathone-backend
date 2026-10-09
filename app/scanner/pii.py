"""Deterministic PII / secret detection with checksum validation.

Design rule: a regex hit is only a *candidate*. Aadhaar must pass Verhoeff, cards must pass
Luhn, PAN must have a valid holder-type letter. This keeps false positives low without
sending any raw data to an LLM.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable

# ---------------------------------------------------------------- checksums

_VD = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VP = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_VINV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_valid(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VD[c][_VP[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(digits: str) -> str:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VD[c][_VP[(i + 1) % 8][int(ch)]]
    return str(_VINV[c])


def luhn_valid(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


# ---------------------------------------------------------------- validators

def is_aadhaar(v: str) -> bool:
    d = _digits(v)
    return len(d) == 12 and d[0] not in "01" and verhoeff_valid(d)


_PAN_RE = re.compile(r"^[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z]$")


def is_pan(v: str) -> bool:
    return bool(_PAN_RE.match(v.strip().upper()))


_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def is_email(v: str) -> bool:
    return bool(_EMAIL_RE.match(v.strip()))


def is_phone_in(v: str) -> bool:
    s = v.strip()
    if re.search(r"[A-Za-z]", s):
        return False
    d = _digits(s)
    if len(d) == 12 and d.startswith("91"):
        d = d[2:]
    elif len(d) == 11 and d.startswith("0"):
        d = d[1:]
    return len(d) == 10 and d[0] in "6789"


def is_phone_intl(v: str) -> bool:
    """A written-out phone number in free text: '+44 1484 682687', '01484 682687', '(022) 2345 6789'.
    Needs a trunk 0 or a +country code, so IDs and counts in prose are not mistaken for phones."""
    s = v.strip()
    d = _digits(s)
    if not 10 <= len(d) <= 13 or len(set(d)) < 4:
        return False
    if s.startswith("+"):
        return True
    # trunk-0 numbers: '01484 682687', '011-2345-6789'; a bare 10-digit run like '0123456789' is more likely an ID
    return s.lstrip("(").startswith("0") and not d.startswith("00") and (len(d) >= 11 or bool(re.search(r"[\s\-()]", s)))


def is_phone_nanp(v: str) -> bool:
    """A North American number written in the usual shape: '(223) 765-4321', '212.555.0199', '001-691-518-9820x5621'.
    Area code and exchange can't start with 0 or 1, which rules out most IDs and dates."""
    d = _digits(re.split(r"(?i)\s*(?:x|ext\.?)\s*\d", v)[0])
    if d.startswith("001"):
        d = d[3:]
    elif len(d) == 11 and d.startswith("1"):
        d = d[1:]
    return len(d) == 10 and d[0] in "23456789" and d[3] in "23456789" and len(set(d)) >= 4


def is_card(v: str) -> bool:
    s = v.strip()
    if re.search(r"[^\d\s\-]", s):
        return False
    d = _digits(s)
    return 13 <= len(d) <= 19 and d[0] in "3456" and luhn_valid(d)


_IFSC_RE = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")


def is_ifsc(v: str) -> bool:
    return bool(_IFSC_RE.match(v.strip().upper()))


_GSTIN_RE = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")


def is_gstin(v: str) -> bool:
    return bool(_GSTIN_RE.match(v.strip().upper()))


_PERSON_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .,'\-]{1,60}$")


def is_person_name(v: str) -> bool:
    s = v.strip()
    return bool(_PERSON_NAME_RE.match(s)) and 1 <= len(s.split()) <= 6


@dataclass(frozen=True)
class EntityType:
    code: str
    label: str
    sensitivity: float          # 0..1, drives risk scoring
    validator: Callable[[str], bool] | None
    name_hints: tuple[str, ...] = ()


ENTITIES: dict[str, EntityType] = {
    e.code: e
    for e in [
        EntityType("AADHAAR", "Aadhaar number", 1.0, is_aadhaar, ("aadhaar", "aadhar", "uidai", "uid_no")),
        EntityType("PAN", "PAN", 0.9, is_pan, ("pan", "pan_no", "pan_number")),
        EntityType("CREDIT_CARD", "Payment card", 1.0, is_card, ("card", "card_no", "cc_number")),
        EntityType("BANK_ACCOUNT", "Bank account", 0.9, None, ("account_no", "account_number", "bank_account", "acct_no")),
        EntityType("PHONE", "Phone number", 0.5, is_phone_in, ("phone", "mobile", "contact_no", "whatsapp")),
        EntityType("EMAIL", "Email address", 0.4, is_email, ("email", "e_mail", "mail")),
        EntityType("DOB", "Date of birth", 0.5, None, ("dob", "date_of_birth", "birth_date", "birthdate")),
        EntityType("PASSWORD", "Password", 1.0, None, ("password", "passwd", "pwd")),
        EntityType("IFSC", "IFSC code", 0.2, is_ifsc, ("ifsc",)),
        EntityType("GSTIN", "GSTIN", 0.2, is_gstin, ("gstin", "gst_no")),
        # Recognised from the column name only: the values ("Cancer", "Hindu", "Rahul Sharma") look like any text.
        EntityType("PERSON_NAME", "Person name", 0.3, is_person_name,
                   ("full_name", "customer_name", "employee_name", "patient_name", "student_name", "first_name", "last_name",
                    "contact_name", "person_name", "holder_name", "card_holder", "applicant_name")),
        EntityType("HEALTH", "Health information", 0.9, None,
                   ("medical_condition", "diagnosis", "diagnoses", "medication", "medications", "prescription", "blood_type",
                    "blood_group", "test_results", "test_result", "disease", "symptoms", "allergies", "allergy",
                    "transcription", "medical_history", "health_condition", "treatment")),
        EntityType("SPECIAL_CATEGORY", "Race, religion or similar", 0.8, None,
                   ("race", "racedesc", "ethnicity", "religion", "caste", "hispaniclatino", "sexual_orientation",
                    "disability", "political_affiliation", "tribe")),
        EntityType("PROFILE_URL", "Social profile link", 0.3, None, ()),   # found inside free text only
    ]
}
NAME_ONLY_ENTITIES = {"PERSON_NAME"}   # a validator alone would call any city or product column a name

# ---------------------------------------------------------------- column classification


def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _hint_match(col: str, hints: Iterable[str]) -> bool:
    tokens = set(col.split("_"))
    return any(h == col or h in tokens or (len(h) > 4 and h in col) for h in hints)


@dataclass
class ColumnPII:
    column: str
    entity: str
    match_ratio: float
    non_null: int
    by_name_only: bool = False


def classify_column(name: str, values: list[str], min_ratio: float = 0.6) -> ColumnPII | None:
    """Return the PII entity of a column, or None. `values` are non-null string samples."""
    col = _norm_col(name)
    vals = [v.strip() for v in values if v is not None and str(v).strip() != ""]
    n = len(vals)
    if n == 0:
        return None
    best: ColumnPII | None = None
    for ent in ENTITIES.values():
        hinted = _hint_match(col, ent.name_hints)
        if ent.code in NAME_ONLY_ENTITIES:
            if not (hinted or col == "name"):
                continue
            ratio = sum(1 for v in vals if ent.validator(v)) / n
            if ratio >= 0.8 and best is None:
                best = ColumnPII(name, ent.code, round(ratio, 3), n, by_name_only=True)
            continue
        if ent.validator is None:
            if not hinted:
                continue
            if ent.code == "BANK_ACCOUNT":
                ok = sum(1 for v in vals if re.fullmatch(r"\d{9,18}", _digits(v) or "x") and not re.search(r"[A-Za-z]", v))
                ratio = ok / n
            else:  # DOB / PASSWORD rely on the column name
                ratio = 1.0
            if ratio >= 0.5:
                cand = ColumnPII(name, ent.code, round(ratio, 3), n, by_name_only=True)
                if best is None or ent.sensitivity > ENTITIES[best.entity].sensitivity:
                    best = cand
            continue
        ok = sum(1 for v in vals if ent.validator(v))
        ratio = ok / n
        threshold = min_ratio * 0.5 if hinted else min_ratio
        if ratio >= threshold:
            cand = ColumnPII(name, ent.code, round(ratio, 3), n)
            if best is None or ratio > best.match_ratio or (
                ratio == best.match_ratio and ent.sensitivity > ENTITIES[best.entity].sensitivity
            ):
                best = cand
    return best


# ---------------------------------------------------------------- free-text scanning

_TEXT_PATTERNS: list[tuple[str, re.Pattern, Callable[[str], bool] | None]] = [
    ("AADHAAR", re.compile(r"(?<!\d)[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}(?!\d)"), is_aadhaar),
    ("PAN", re.compile(r"(?<![A-Z0-9])[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z](?![A-Z0-9])"), is_pan),
    ("CREDIT_CARD", re.compile(r"(?<!\d)(?:\d[ \-]?){12,18}\d(?!\d)"), is_card),
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), None),
    ("PHONE", re.compile(r"(?<![\d+])(?:\+91[\-\s]?|0)?[6-9]\d{9}(?!\d)"), is_phone_in),
    # other written-out numbers (landlines, foreign mobiles); spaces/dashes only between digit groups
    ("PHONE", re.compile(r"(?<![\w+])(?:\+\d{1,3}[ \-]?(?:\(\d{1,4}\)[ \-]?)?|\(0\d{1,4}\)[ \-]?|0)\d{2,5}(?:[ \-]?\d{2,5}){1,3}(?![\w])"),
     is_phone_intl),
    # North American layout; the separators are required, so a bare 10-digit run (an order or tweet ID) never matches
    ("PHONE", re.compile(r"(?<![\w+])(?:\+?1[ .\-]?|001[ .\-])?(?:\(\d{3}\)[ .\-]?|\d{3}[.\-])\d{3}[.\-]\d{4}(?:\s?(?:x|ext\.?)\s?\d{1,5})?(?![\w])", re.I),
     is_phone_nanp),
]

SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("API secret key", re.compile(r"\bsk-(?:ant-|live-|proj-)?[A-Za-z0-9_\-]{20,}")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("Razorpay key", re.compile(r"\brzp_(?:live|test)_[A-Za-z0-9]{14}\b")),
    ("Private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("Credential in config", re.compile(
        r"(?im)^[ \t]*(?:export[ \t]+)?[A-Z0-9_.-]*(?:PASSWORD|PASSWD|PWD|SECRET|TOKEN|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY)"
        r"[A-Z0-9_.-]*[ \t]*[:=][ \t]*['\"]?[^\s'\"]{6,}")),
]


@dataclass
class TextScan:
    entities: Counter = field(default_factory=Counter)
    secrets: Counter = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return sum(self.entities.values())


_URL_SPAN = re.compile(r"(?:https?://|www\.)\S+|(?<![\w./])(?:linkedin|instagram|facebook|twitter|github)\.com/\S+", re.I)
# a link to one person's account; plain site links (news, shops, t.co) are not personal data
_PROFILE_URL = re.compile(
    r"(?:https?://)?(?:www\.|[a-z]{2}\.)?(?:linkedin\.com/in|instagram\.com|facebook\.com|twitter\.com|x\.com|github\.com"
    r"|youtube\.com/(?:c|channel|user|@)|t\.me)/[\w.\-@]{2,}", re.I)


# about 1 in 10 random 12-digit numbers (order IDs, essay ID numbers) passes Verhoeff, so in prose an Aadhaar
# is only counted when it is introduced as one ("Aadhaar 2345 ...", "UID no: ...")
_AADHAAR_CONTEXT = re.compile(r"(?i)aadh?aa?r|\buid(?:ai)?\b|\bunique id")


def _text_hits(text: str) -> list[tuple[str, int, int]]:
    """Validated (entity, start, end) spans; overlapping spans keep the earlier, more specific pattern.
    Digits inside links (pinterest.com/pin/4469119629...) are IDs, not cards or phones."""
    hits: list[tuple[str, int, int]] = []
    if "/" in text or "www" in text:
        for m in _URL_SPAN.finditer(text):
            url = m.group(0).rstrip(".,;:!?)]}'\"")   # "see https://x.com/a, or" - the comma isn't part of the link
            hits.append(("PROFILE_URL" if _PROFILE_URL.match(url) else "URL", m.start(), m.start() + len(url)))
    for code, rx, validator in _TEXT_PATTERNS:
        for m in rx.finditer(text):
            if code == "AADHAAR" and not _AADHAAR_CONTEXT.search(text[max(0, m.start() - 40):m.start()]):
                continue
            if (validator is None or validator(m.group(0))) and not any(m.start() < e and s < m.end() for _, s, e in hits):
                hits.append((code, m.start(), m.end()))
    return [h for h in hits if h[0] != "URL"]


def scan_text(text: str, secrets: bool = True) -> TextScan:
    res = TextScan()
    for code, _, _ in _text_hits(text):
        res.entities[code] += 1
    if not secrets:
        return res
    # count credentials per line so "AWS_ACCESS_KEY_ID=AKIA..." is one secret, not two
    for line in text.splitlines():
        labels = [label for label, rx in SECRET_PATTERNS if rx.search(line)]
        if labels:
            specific = [l for l in labels if l != "Credential in config"]
            res.secrets[specific[0] if specific else labels[0]] += 1
    return res


# ---------------------------------------------------------------- masking


def mask_value(entity: str, value: str) -> str:
    v = str(value)
    if entity == "EMAIL" and "@" in v:
        local, _, dom = v.partition("@")
        return (local[:1] + "***@" + dom) if local else "***@" + dom
    if entity in ("AADHAAR", "PHONE", "CREDIT_CARD", "BANK_ACCOUNT"):
        d = _digits(v)
        return "X" * max(0, len(d) - 4) + d[-4:]
    if entity == "PAN":
        return "XXXXX" + v[5:9] + "X" if len(v) >= 10 else "XXXXXXXXXX"
    if entity in ("PASSWORD", "DOB"):
        return "••••••"
    if entity == "PROFILE_URL":
        m = re.match(r"(?:https?://)?(?:www\.)?([^/\s]+/(?:in/|c/|channel/|user/)?)", v)
        return (m.group(1) if m else "") + "***"
    return v[:2] + "***" if len(v) > 2 else "***"


def mask_text(text: str) -> str:
    """Redact every validated detection; used before any text reaches an LLM."""
    out = text
    for label, rx in SECRET_PATTERNS:
        out = rx.sub(f"[{label.upper()} REDACTED]", out)
    for code, s, e in sorted(_text_hits(out), key=lambda h: -h[1]):
        out = out[:s] + mask_value(code, out[s:e]) + out[e:]
    return out
