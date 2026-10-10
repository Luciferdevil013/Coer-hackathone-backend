"""Deterministic column profiling and data-quality issue detection (FR-04)."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from app.scanner import pii, textnorm

_DATE_PATTERNS = [
    ("YYYY-MM-DD", re.compile(r"^\d{4}-\d{2}-\d{2}$")),
    ("DD/MM/YYYY", re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")),
    ("DD-MM-YYYY", re.compile(r"^\d{1,2}-\d{1,2}-\d{4}$")),
    ("YYYY/MM/DD", re.compile(r"^\d{4}/\d{2}/\d{2}$")),
    ("DD Mon YYYY", re.compile(r"^\d{1,2} [A-Za-z]{3,9} \d{4}$")),
    ("DD-Mon-YYYY", re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{4}$")),
    ("YYYY-MM-DD HH:MM", re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?$")),
]
_INT = re.compile(r"^-?\d+$")
_DEC = re.compile(r"^-?\d+(?:,\d{3})*\.\d+$")
_BOOL = {"true", "false", "yes", "no", "y", "n"}
CANONICAL_PHONE = re.compile(r"^\+91[6-9]\d{9}$")
FREE_TEXT_MIN_CHARS = 25
NEAR_EMPTY_RATE = 0.98    # a column with values in under 2% of rows carries almost no information
PROSE_ENTITIES = (None, "HEALTH", "SPECIAL_CATEGORY")   # a doctor's note is health data *and* free text
# U+FFFD, UTF-8 read as Latin-1 ('Ã©', 'â€™'), or a run of '?' where characters were lost on export
_MOJIBAKE = re.compile(r"�|Ã[\u0080-¿]|â€|Â[ -¿ ]")
_LOST_CHARS = re.compile(r"\?{3,}")
_PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]{1,40}\}")
_MONEY_COL = re.compile(r"(^|_)(amount|amt|price|salary|fee|fees|billing|bill|cost|income|payment|revenue|wage)s?($|_)")
_AGE_COL = re.compile(r"(^|_)age($|_)")
_EXPORT_INDEX_COL = re.compile(r"^(unnamed_?\d+|index|level_0)$")
_NAME_PREFIX_OK = re.compile(r"^(Mc|Mac|O'|De|Di|Da|La|Le|Van|Von|St)[A-Z]")
_EVENT_WORDS = re.compile(
    r"(^|_)(tickets?|orders?|transactions?|txns?|invoices?|bookings?|payments?|visits?|appointments?|"
    r"admissions?|claims?|calls?|sessions?|events?|logs?|shipments?|requests?|cases?|incidents?|"
    r"reviews?|ratings?|rates?|feedbacks?|comments?|opinions?|sentiments?)($|_)"
)


def is_event_dataset(view: DatasetView) -> bool:
    """True when rows represent events/feedback (reviews, orders, visits, logs) rather than unique entities."""
    for c in view.df.columns:
        if _EVENT_WORDS.search(norm_col(c)):
            return True
    lbl = (getattr(view, "label", None) or "").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return bool(_EVENT_WORDS.search(norm_col(lbl)))


def _odd_capitals(name: str) -> bool:
    """'JacksOn' or 'leSLie': capitals inside a word that no naming convention explains (McDonald and DeSouza are fine)."""
    for w in re.split(r"[\s,]+", name):
        if len(w) > 1 and w.isalpha() and w not in (w.title(), w.upper(), w.lower()) and not _NAME_PREFIX_OK.match(w):
            return True
    return False


def _to_float(v: str) -> float | None:
    try:
        return float(v.replace(",", ""))
    except ValueError:
        return None


def date_format(v: str) -> str | None:
    for label, rx in _DATE_PATTERNS:
        if rx.match(v):
            return label
    return None


@dataclass
class ColumnProfile:
    name: str
    inferred_type: str
    non_null: int
    nulls: int
    null_rate: float
    distinct: int
    unique_ratio: float
    whitespace_issues: int = 0
    case_issues: int = 0
    invalid_values: int = 0
    format_variants: dict = field(default_factory=dict)
    non_canonical: int = 0
    pii_entity: str | None = None
    pii_ratio: float = 0.0
    avg_length: float = 0.0
    free_text: bool = False         # long, varied prose (tweets, comments, notes) rather than a field
    html_entities: int = 0          # values with HTML codes such as &amp; instead of "&"
    text_pii: dict = field(default_factory=dict)   # {entity: count} found *inside* free text
    text_pii_rows: int = 0
    text_pii_row_ids: set = field(default_factory=set, repr=False)   # row positions, so a file counts each row once
    garbled: int = 0                # broken characters from an encoding mix-up ('Ã©', '??????')
    placeholders: dict = field(default_factory=dict)   # unfilled template fields, e.g. {"{product_purchased}": 8469}
    name_case_issues: int = 0       # 'Bobby JacksOn'
    negative_values: int = 0        # negative amounts in a money column
    impossible_ages: int = 0        # age below 0 or above 120
    export_index: bool = False      # 'Unnamed: 0' row counter left behind by a spreadsheet/pandas export
    samples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("text_pii_row_ids", None)
        return d


@dataclass
class QualityIssue:
    code: str
    column: str | None
    count: int
    message: str
    suggested_op: str | None = None


@dataclass
class DatasetView:
    """In-memory, scan-scoped view of one tabular asset (or one database table)."""
    asset_id: str
    label: str
    df: pd.DataFrame
    profiles: dict[str, ColumnProfile] = field(default_factory=dict)
    issues: list[QualityIssue] = field(default_factory=list)
    exact_duplicate_rows: int = 0
    text_duplicates: dict = field(default_factory=dict)   # set by the scanner's text-duplicate detector
    _hash_cache: dict = field(default_factory=dict, repr=False)

    @property
    def rows(self) -> int:
        return len(self.df)

    @property
    def norm_columns(self) -> dict[str, str]:
        return {norm_col(c): c for c in self.df.columns}

    def pii_columns(self) -> dict[str, str]:
        return {c: p.pii_entity for c, p in self.profiles.items() if p.pii_entity}

    def row_hashes(self, cols: tuple[str, ...]) -> set[int]:
        """Hashes of normalised rows restricted to `cols` (actual column names, in a fixed order) -
        used for cross-dataset similarity, so two files with renamed headers still compare."""
        if cols not in self._hash_cache:
            sub = self.df[list(cols)].fillna("").astype(str)
            sub = sub.apply(lambda s: s.str.strip().str.lower())
            sub.columns = range(len(cols))
            self._hash_cache[cols] = set(pd.util.hash_pandas_object(sub, index=False).tolist())
        return self._hash_cache[cols]


def norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _infer_type(vals: list[str]) -> str:
    if not vals:
        return "empty"
    n = len(vals)
    counts = Counter()
    for v in vals:
        if _INT.match(v):
            counts["integer"] += 1
        elif _DEC.match(v):
            counts["decimal"] += 1
        elif v.lower() in _BOOL:
            counts["boolean"] += 1
        elif date_format(v):
            counts["date"] += 1
        elif pii.is_email(v):
            counts["email"] += 1
        elif pii.is_phone_in(v):
            counts["phone"] += 1
        else:
            counts["text"] += 1
    if counts.get("phone", 0) + counts.get("integer", 0) == n and counts.get("phone", 0) >= 0.8 * n:
        return "phone"
    top, c = counts.most_common(1)[0]
    if c >= 0.8 * n:
        return top
    return "mixed"


def looks_like_prose(stripped: list[str], unique_ratio: float, inferred_type: str) -> bool:
    """Long, varied text (notes, tweets, comments) rather than a field. Shared with the trim step, so the
    whitespace the scan reports is exactly what cleaning changes."""
    if inferred_type not in ("text", "mixed") or len(stripped) < 5:
        return False
    if sum(1 for v in stripped[:500] if v[:1] in "[{") >= 0.8 * min(len(stripped), 500):
        return False   # serialized lists/objects ("['Tiburce', 'Evans', ...]") are data, not prose
    avg = sum(len(v) for v in stripped) / len(stripped)
    return avg >= FREE_TEXT_MIN_CHARS and unique_ratio >= 0.3


def prose_column(values: list) -> bool:
    stripped = [str(v).strip() for v in values if v is not None and str(v).strip() and str(v) != "nan"]
    if not stripped:
        return False
    sample = stripped[:5000]
    return looks_like_prose(stripped, len(set(stripped)) / len(stripped), _infer_type(sample))


def _text_hygiene(prof: ColumnProfile, col: str, stripped: list[str], n_rows: int) -> None:
    """Checks that need the column's meaning: broken characters, template leftovers, name casing, impossible numbers."""
    name = norm_col(col)
    if prof.inferred_type in ("text", "mixed"):
        # '???' is normal emphasis in a tweet, but in a product name or city it means characters were lost
        prof.garbled = sum(1 for v in stripped if _MOJIBAKE.search(v) or (not prof.free_text and _LOST_CHARS.search(v)))
        found = Counter(m for v in stripped for m in set(_PLACEHOLDER.findall(v)))
        rows_with = sum(1 for v in stripped if "{" in v and _PLACEHOLDER.search(v))
        # one stray '{IMMUNOASSAY}' in a note is noise; a template leftover repeats
        if found and rows_with >= max(5, 0.01 * len(stripped)):
            prof.placeholders = {"rows": rows_with, **dict(found.most_common(5))}
    if prof.pii_entity == "PERSON_NAME":
        prof.name_case_issues = sum(1 for v in stripped if _odd_capitals(v))
    if prof.inferred_type in ("integer", "decimal"):
        nums = [x for x in (_to_float(v) for v in stripped) if x is not None]
        if _MONEY_COL.search(name):
            prof.negative_values = sum(1 for x in nums if x < 0)
        if _AGE_COL.search(name):
            prof.impossible_ages = sum(1 for x in nums if x < 0 or x > 120)
        if _EXPORT_INDEX_COL.match(name) and prof.inferred_type == "integer" and len(nums) == n_rows > 1:
            ordered = sorted(nums)
            prof.export_index = ordered[0] in (0, 1) and ordered[-1] - ordered[0] == n_rows - 1


def profile_dataframe(view: DatasetView, sample_limit: int = 5000, pii_ratio: float = 0.6) -> None:
    df = view.df
    for col in df.columns:
        series = df[col]
        non_null_series = series.dropna()
        non_null_series = non_null_series[non_null_series.astype(str).str.strip() != ""]
        vals = non_null_series.astype(str).tolist()
        non_null_pos = series.index.get_indexer(non_null_series.index).tolist()
        n_total = len(series)
        nn = len(vals)
        distinct = int(non_null_series.nunique()) if nn else 0
        sample = vals[:sample_limit]
        prof = ColumnProfile(
            name=str(col),
            inferred_type=_infer_type([v.strip() for v in sample]),
            non_null=nn,
            nulls=n_total - nn,
            null_rate=round((n_total - nn) / n_total, 4) if n_total else 0.0,
            distinct=distinct,
            unique_ratio=round(distinct / nn, 4) if nn else 0.0,
        )
        prof.whitespace_issues = sum(1 for v in vals if v != v.strip() or "  " in v)
        cls = pii.classify_column(str(col), sample, pii_ratio)
        if cls:
            prof.pii_entity, prof.pii_ratio = cls.entity, cls.match_ratio
        stripped = [v.strip() for v in vals]
        if prof.pii_entity == "EMAIL" or prof.inferred_type == "email":
            prof.case_issues = sum(1 for v in stripped if v != v.lower())
            prof.invalid_values = sum(1 for v in stripped if not pii.is_email(v))
        if prof.pii_entity == "PHONE" or prof.inferred_type == "phone":
            prof.invalid_values = sum(1 for v in stripped if not pii.is_phone_in(v))
            valid = [v for v in stripped if pii.is_phone_in(v)]
            shapes = Counter(re.sub(r"\d", "9", v) for v in valid)
            # a single consistent format is fine; mixed formats break joins and dedupe
            if valid and shapes.most_common(1)[0][1] < 0.98 * len(valid):
                prof.non_canonical = sum(1 for v in valid if not CANONICAL_PHONE.match(v))
                prof.format_variants = dict(shapes.most_common(6))
        if prof.inferred_type in ("date", "mixed"):
            fmts = Counter(f for f in (date_format(v) for v in stripped) if f)
            if fmts and sum(fmts.values()) >= 0.8 * max(nn, 1):
                prof.inferred_type = "date"
                prof.format_variants = dict(fmts)
                dominant = fmts.most_common(1)[0][1]
                prof.non_canonical = sum(fmts.values()) - dominant
        if stripped:
            prof.avg_length = round(sum(len(v) for v in stripped) / len(stripped), 1)
        if prof.inferred_type in ("text", "mixed") and prof.pii_entity in PROSE_ENTITIES:
            prof.free_text = looks_like_prose(stripped, prof.unique_ratio, prof.inferred_type)
            prof.html_entities = sum(1 for v in vals if "&" in v and textnorm.has_html_entities(v))
        _text_hygiene(prof, str(col), stripped, len(df))
        if prof.free_text:
            # double spaces inside a paragraph are normal prose; only padding around the value is a defect
            prof.whitespace_issues = sum(1 for v in vals if v != v.strip())
            counts: Counter = Counter()
            for pos, v in zip(non_null_pos, vals):
                found = pii.scan_text(v, secrets=False).entities
                if found:
                    counts.update(found)
                    prof.text_pii_rows += 1
                    prof.text_pii_row_ids.add(pos)
            prof.text_pii = dict(counts)
            # a pasted value can't be shown in the clear, even in the column profile
            prof.samples = [pii.mask_text(v)[:40] for v in stripped[:3]]
        prof.samples = prof.samples or [pii.mask_value(prof.pii_entity, v) if prof.pii_entity else v[:40] for v in stripped[:3]]
        view.profiles[str(col)] = prof

    # exact duplicate rows (after trimming whitespace)
    if len(df):
        trimmed = df.fillna("").astype(str).apply(lambda s: s.str.strip())
        view.exact_duplicate_rows = int(trimmed.duplicated().sum())
    _derive_issues(view)


def _derive_issues(view: DatasetView) -> None:
    issues: list[QualityIssue] = []
    for col, p in view.profiles.items():
        if p.whitespace_issues:
            issues.append(QualityIssue("whitespace", col, p.whitespace_issues,
                                       f"{p.whitespace_issues:,} values in '{col}' have stray spaces", "trim_whitespace"))
        if p.case_issues:
            issues.append(QualityIssue("email_case", col, p.case_issues,
                                       f"{p.case_issues:,} emails in '{col}' are not lowercase", "normalize_email"))
        if p.invalid_values:
            issues.append(QualityIssue("invalid_format", col, p.invalid_values,
                                       f"{p.invalid_values:,} values in '{col}' fail {p.pii_entity or p.inferred_type} validation",
                                       "validate_rules"))
        if p.pii_entity == "PHONE" and p.non_canonical:
            issues.append(QualityIssue("phone_format", col, p.non_canonical,
                                       f"{p.non_canonical:,} phone numbers in '{col}' are not in +91XXXXXXXXXX format",
                                       "normalize_phone"))
        if p.inferred_type == "date" and p.non_canonical:
            issues.append(QualityIssue("date_format", col, p.non_canonical,
                                       f"'{col}' mixes {len(p.format_variants)} date formats ({p.non_canonical:,} non-standard)",
                                       "parse_dates"))
        if p.pii_entity in ("EMAIL", "PHONE") and p.nulls and p.null_rate >= 0.01:
            issues.append(QualityIssue("missing_key", col, p.nulls,
                                       f"{p.nulls:,} rows are missing '{col}'", "validate_rules"))
        if p.html_entities:
            issues.append(QualityIssue("html_entities", col, p.html_entities,
                                       f"{p.html_entities:,} values in '{col}' contain HTML codes such as &amp; instead of the character",
                                       "decode_html"))
        if p.garbled:
            issues.append(QualityIssue("garbled_text", col, p.garbled,
                                       f"{p.garbled:,} values in '{col}' contain garbled characters (such as 'Ã©' or '??????') "
                                       "from a text-encoding mix-up", "repair_text"))
        if p.placeholders:
            rows_with = p.placeholders["rows"]
            top, cnt = next((k, v) for k, v in p.placeholders.items() if k != "rows")
            issues.append(QualityIssue("template_placeholder", col, rows_with,
                                       f"{rows_with:,} values in '{col}' still contain unfilled template fields such as {top} "
                                       f"({cnt:,} times)"))
        if p.name_case_issues:
            issues.append(QualityIssue("name_case", col, p.name_case_issues,
                                       f"{p.name_case_issues:,} names in '{col}' have random capital letters (for example "
                                       "'JacksOn' instead of 'Jackson')", "standardize_case"))
        if p.negative_values:
            issues.append(QualityIssue("negative_amount", col, p.negative_values,
                                       f"{p.negative_values:,} values in '{col}' are negative amounts", "validate_rules"))
        if p.impossible_ages:
            issues.append(QualityIssue("impossible_age", col, p.impossible_ages,
                                       f"{p.impossible_ages:,} values in '{col}' are impossible ages (below 0 or above 120)",
                                       "validate_rules"))
    n = view.rows
    for col, p in view.profiles.items():
        if p.export_index:
            issues.append(QualityIssue("export_index", col, 1,
                                       f"Column '{col}' is a leftover row number from a spreadsheet export; consider dropping it",
                                       "drop_columns"))
        if n >= 50 and p.null_rate >= NEAR_EMPTY_RATE:
            detail = "has no values" if not p.non_null else f"is {p.null_rate:.2%} empty ({p.non_null:,} value{'s' if p.non_null != 1 else ''} in {n:,} rows)"
            issues.append(QualityIssue("empty_column", col, 1, f"Column '{col}' {detail}; consider dropping it", "drop_columns"))
    if view.exact_duplicate_rows:
        if is_event_dataset(view):
            issues.append(QualityIssue("duplicate_rows", None, view.exact_duplicate_rows,
                                       f"{view.exact_duplicate_rows:,} rows share identical ratings/feedback (common for review and event logs from multiple users; preserved as separate entries)",
                                       None))
        else:
            issues.append(QualityIssue("duplicate_rows", None, view.exact_duplicate_rows,
                                       f"{view.exact_duplicate_rows:,} rows are exact duplicates", "deduplicate"))
    view.issues = issues
