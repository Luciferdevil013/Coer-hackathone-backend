"""Deterministic, tested operator implementations (the only code that changes data).

Every operator works on an all-string DataFrame carrying a hidden `__row_id` column so
changes, merges and quarantines can always be traced back to the original row.
"""
from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from app.pipeline import spec as S
from app.scanner import pii, textnorm
from app.scanner.profiler import date_format

ROW_ID = "__row_id"
REASON = "__reason"


@dataclass
class OpResult:
    df: pd.DataFrame
    quarantined: pd.DataFrame | None = None
    cells_changed: int = 0
    rows_removed: int = 0
    changes: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    removed: dict[int, int] = field(default_factory=dict)  # merged-away row id -> surviving row id


def _empty(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and v.strip() == "")


def _text_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    if cols == ["*"] or "*" in cols:
        return [c for c in df.columns if c != ROW_ID]
    return cols


def _map_cells(df: pd.DataFrame, cols: list[str], fn) -> OpResult:
    out = df.copy()
    changed = 0
    changes = []
    for c in cols:
        before = out[c].tolist()
        after = [v if _empty(v) else fn(v) for v in before]
        for i, (b, a) in enumerate(zip(before, after)):
            if b != a and not (_empty(b) and _empty(a)):
                changed += 1
                if len(changes) < 40:
                    changes.append({"row": int(out[ROW_ID].iat[i]), "column": c, "before": b, "after": a})
        out[c] = after
    return OpResult(out, cells_changed=changed, changes=changes)


# ---------------------------------------------------------------- cell operators


def trim_whitespace(df, p: S.ColumnsParams, _ctx) -> OpResult:
    """Fields lose stray and doubled spaces; prose (notes, tweets) only loses padding at the ends,
    matching what the scan counts as a whitespace issue."""
    from app.scanner.profiler import prose_column

    res_df, total, changes = df, 0, []
    for c in _text_columns(df, p.columns):
        prose = prose_column(df[c].tolist())
        fn = (lambda v: str(v).strip()) if prose else (lambda v: re.sub(r"\s{2,}", " ", str(v)).strip())
        r = _map_cells(res_df, [c], fn)
        res_df, total = r.df, total + r.cells_changed
        changes.extend(r.changes[: max(0, 40 - len(changes))])
    return OpResult(res_df, cells_changed=total, changes=changes)


def standardize_case(df, p: S.CaseParams, _ctx) -> OpResult:
    fn = {"lower": str.lower, "upper": str.upper, "title": str.title}[p.mode]
    return _map_cells(df, p.columns, lambda v: fn(str(v)))


def normalize_email(df, p: S.ColumnsParams, _ctx) -> OpResult:
    return _map_cells(df, _text_columns(df, p.columns), lambda v: str(v).strip().lower())


def to_e164_in(v: str) -> str:
    s = str(v).strip()
    d = re.sub(r"\D", "", s)
    if len(d) == 12 and d.startswith("91"):
        d = d[2:]
    elif len(d) == 11 and d.startswith("0"):
        d = d[1:]
    if len(d) == 10 and d[0] in "6789" and not re.search(r"[A-Za-z]", s):
        return "+91" + d
    return s


def normalize_phone(df, p: S.PhoneParams, _ctx) -> OpResult:
    return _map_cells(df, p.columns, to_e164_in)


_STRPTIME = {
    "YYYY-MM-DD": ["%Y-%m-%d"],
    "DD/MM/YYYY": ["%d/%m/%Y", "%m/%d/%Y"],
    "DD-MM-YYYY": ["%d-%m-%Y", "%m-%d-%Y"],
    "YYYY/MM/DD": ["%Y/%m/%d"],
    "DD Mon YYYY": ["%d %b %Y", "%d %B %Y"],
    "DD-Mon-YYYY": ["%d-%b-%Y"],
    "YYYY-MM-DD HH:MM": ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"],
}


def parse_date(v: str, dayfirst: bool = True) -> datetime | None:
    s = str(v).strip()
    fmt = date_format(s)
    if not fmt:
        return None
    candidates = _STRPTIME[fmt] if dayfirst else list(reversed(_STRPTIME[fmt]))
    for f in candidates:
        try:
            return datetime.strptime(s, f)
        except ValueError:
            continue
    return None


def parse_dates(df, p: S.DateParams, _ctx) -> OpResult:
    def fn(v):
        d = parse_date(v, p.dayfirst)
        return d.strftime(p.output_format) if d else v

    res = _map_cells(df, p.columns, fn)
    unparsed = sum(1 for c in p.columns for v in res.df[c] if not _empty(v) and parse_date(v) is None)
    if unparsed:
        res.notes.append(f"{unparsed} values could not be parsed as dates and were left unchanged")
    return res


def fill_missing(df, p: S.FillParams, _ctx) -> OpResult:
    out = df.copy()
    changed, changes = 0, []
    for c in p.columns:
        vals = out[c].tolist()
        for i, v in enumerate(vals):
            if _empty(v):
                vals[i] = p.value
                changed += 1
                if len(changes) < 40:
                    changes.append({"row": int(out[ROW_ID].iat[i]), "column": c, "before": "", "after": p.value})
        out[c] = vals
    return OpResult(out, cells_changed=changed, changes=changes)


def mask_pii(df, p: S.MaskParams, _ctx) -> OpResult:
    res_df, total, changes = df, 0, []
    for c in p.columns:
        sample = [str(v) for v in df[c].tolist() if not _empty(v)][:500]
        cls = pii.classify_column(c, sample)
        ent = cls.entity if cls else ""
        if p.method == "hash":
            fn = lambda v: "h_" + hashlib.sha256(str(v).encode()).hexdigest()[:16]  # noqa: E731
        elif p.method == "redact":
            fn = lambda v: "[REDACTED]"  # noqa: E731
        else:
            fn = (lambda v, e=ent: pii.mask_value(e, str(v)))
        r = _map_cells(res_df, [c], fn)
        res_df, total = r.df, total + r.cells_changed
        changes.extend(r.changes[: max(0, 40 - len(changes))])
    return OpResult(res_df, cells_changed=total, changes=changes)


def decode_html(df, p: S.ColumnsParams, _ctx) -> OpResult:
    return _map_cells(df, _text_columns(df, p.columns), lambda v: html.unescape(str(v)) if "&" in str(v) else v)


def repair_text(df, p: S.ColumnsParams, _ctx) -> OpResult:
    """Fields and prose lose garbled characters, mojibake (Ã©) and lost characters (??????)."""
    from app.scanner.profiler import prose_column

    res_df, total, changes = df, 0, []
    for c in _text_columns(df, p.columns):
        prose = prose_column(df[c].tolist())
        fn = (lambda v, pr=prose: textnorm.repair_text(str(v), prose=pr))
        r = _map_cells(res_df, [c], fn)
        res_df, total = r.df, total + r.cells_changed
        changes.extend(r.changes[: max(0, 40 - len(changes))])
    return OpResult(res_df, cells_changed=total, changes=changes)


def redact_text_pii(df, p: S.ColumnsParams, _ctx) -> OpResult:
    return _map_cells(df, _text_columns(df, p.columns), lambda v: pii.mask_text(str(v)))


def drop_columns(df, p: S.ColumnsParams, _ctx) -> OpResult:
    cols = [c for c in p.columns if c in df.columns and c != ROW_ID]
    return OpResult(df.drop(columns=cols), notes=[f"Dropped columns: {', '.join(cols)}"])


# ---------------------------------------------------------------- dedupe / merge


def _key_norm(col: str, v) -> str:
    if _empty(v):
        return ""
    s = str(v).strip().lower()
    if re.search(r"phone|mobile|contact", col.lower()):
        d = re.sub(r"\D", "", s)
        return d[-10:] if len(d) >= 10 else s
    return re.sub(r"\s+", " ", s)


def _completeness(row: pd.Series) -> int:
    return sum(0 if _empty(v) else 1 for k, v in row.items() if k != ROW_ID)


def _resolve_group(df: pd.DataFrame, idxs: list[int], keep: str, conflict_cols: list[str]):
    """Return (survivor_series | None, conflict_columns_found)."""
    rows = df.loc[idxs]
    conflicts = []
    for c in conflict_cols:
        vals = {str(v).strip().upper() for v in rows[c].tolist() if not _empty(v)}
        if len(vals) > 1:
            conflicts.append(c)
    if conflicts:
        return None, conflicts
    if keep == "first":
        order = idxs
    elif keep == "last":
        order = list(reversed(idxs))
    else:
        order = sorted(idxs, key=lambda i: (-_completeness(df.loc[i]), idxs.index(i)))
    survivor = df.loc[order[0]].copy()
    for i in order[1:]:  # coalesce: fill survivor gaps from the other records
        for c in df.columns:
            if c != ROW_ID and _empty(survivor[c]) and not _empty(df.at[i, c]):
                survivor[c] = df.at[i, c]
    return survivor, []


def _apply_groups(df: pd.DataFrame, groups: list[list[int]], keep: str, conflict_cols: list[str], code: str) -> OpResult:
    drop, quarantine_idx, merged_examples = set(), [], []
    removed: dict[int, int] = {}
    reasons: dict[int, str] = {}
    updates: dict[int, pd.Series] = {}
    for idxs in groups:
        if len(idxs) < 2:
            continue
        survivor, conflicts = _resolve_group(df, idxs, keep, conflict_cols)
        if survivor is None:
            for i in idxs:
                quarantine_idx.append(i)
                reasons[i] = f"{code}_CONFLICT: {', '.join(conflicts)} differ"
            continue
        keep_idx = [i for i in idxs if df.at[i, ROW_ID] == survivor[ROW_ID]][0]
        if not survivor.equals(df.loc[keep_idx]):  # only coalesced survivors need writing back
            updates[keep_idx] = survivor
        drop.update(i for i in idxs if i != keep_idx)
        removed.update({int(df.at[i, ROW_ID]): int(survivor[ROW_ID]) for i in idxs if i != keep_idx})
        if len(merged_examples) < 15:
            merged_examples.append({"kept_row": int(survivor[ROW_ID]),
                                    "merged_rows": [int(df.at[i, ROW_ID]) for i in idxs if i != keep_idx]})
    out = df.copy()
    if updates:  # one bulk write; per-row .loc assignment is O(rows x columns) block copies in pandas
        idx = list(updates)
        out.loc[idx, list(out.columns)] = pd.DataFrame([updates[i] for i in idx], index=idx)[list(out.columns)].to_numpy()
    q = out.loc[quarantine_idx].copy() if quarantine_idx else None
    if q is not None:
        q[REASON] = [reasons[i] for i in quarantine_idx]
    out = out.drop(index=list(drop | set(quarantine_idx)))
    res = OpResult(out, quarantined=q, rows_removed=len(drop), removed=removed)
    res.changes = [{"row": m["kept_row"], "column": "(record)", "before": f"{len(m['merged_rows']) + 1} records",
                    "after": f"merged rows {m['merged_rows']} into {m['kept_row']}"} for m in merged_examples]
    if quarantine_idx:
        res.notes.append(f"{len(quarantine_idx)} rows quarantined: duplicates with conflicting {', '.join(conflict_cols)}")
    return res


def deduplicate(df, p: S.DedupeParams, _ctx) -> OpResult:
    df = df.reset_index(drop=True)
    norm = (lambda c, v: textnorm.content(v)) if p.text_match == "content" else _key_norm
    normed = [[norm(c, v) for v in df[c].tolist()] for c in p.keys]
    groups: dict[tuple, list[int]] = {}
    for i, k in enumerate(zip(*normed)):
        if all(x == "" for x in k):
            continue  # never group rows that have no key value
        groups.setdefault(k, []).append(i)
    res = _apply_groups(df, list(groups.values()), p.keep, p.conflict_columns, "DEDUP")
    how = "; links, retweet prefixes and punctuation ignored" if p.text_match == "content" else ""
    res.notes.insert(0, f"{res.rows_removed} duplicate rows merged into their survivors (keys: {', '.join(p.keys)}; keep: {p.keep}{how})")
    return res


def merge_rows(df, p: S.MergeParams, _ctx) -> OpResult:
    df = df.reset_index(drop=True)
    pos = {int(r): i for i, r in enumerate(df[ROW_ID].tolist())}
    groups = [[pos[r] for r in cl if r in pos] for cl in p.clusters]
    res = _apply_groups(df, groups, p.keep, p.conflict_columns, "MERGE")
    res.notes.insert(0, f"{res.rows_removed} rows merged from {len(p.clusters)} reviewed duplicate groups")
    return res


# ---------------------------------------------------------------- validation

_FORMAT_FN = {
    "email": pii.is_email,
    "phone": pii.is_phone_in,
    "pan": pii.is_pan,
    "aadhaar": pii.is_aadhaar,
    "ifsc": pii.is_ifsc,
    "date": lambda v: date_format(str(v).strip()) is not None,
}


def validate_rules(df, p: S.ValidateParams, _ctx) -> OpResult:
    df = df.reset_index(drop=True)
    reasons: dict[int, list[str]] = {}
    for c in p.required:
        for i, v in enumerate(df[c].tolist()):
            if _empty(v):
                reasons.setdefault(i, []).append(f"REQUIRED_MISSING: {c}")
    for c, fmt in p.formats.items():
        fn = _FORMAT_FN[fmt]
        for i, v in enumerate(df[c].tolist()):
            if not _empty(v) and not fn(str(v)):
                reasons.setdefault(i, []).append(f"INVALID_FORMAT: {c} ({fmt})")
    for c, allowed in p.allowed_values.items():
        allowed_l = {a.lower() for a in allowed}
        for i, v in enumerate(df[c].tolist()):
            if not _empty(v) and str(v).strip().lower() not in allowed_l:
                reasons.setdefault(i, []).append(f"NOT_ALLOWED: {c}={v}")
    for c in p.unique:
        seen = set()
        for i, v in enumerate(df[c].tolist()):
            k = _key_norm(c, v)
            if not k:
                continue
            if k in seen:
                reasons.setdefault(i, []).append(f"DUPLICATE_KEY: {c}")
            seen.add(k)
    bad = sorted(reasons)
    q = df.loc[bad].copy() if bad else None
    if q is not None:
        q[REASON] = ["; ".join(reasons[i]) for i in bad]
    out = df.drop(index=bad)
    res = OpResult(out, quarantined=q)
    if bad:
        res.notes.append(f"{len(bad)} rows failed validation and were quarantined")
    return res


def quarantine_anomalies(df, p: S.AnomalyParams, _ctx) -> OpResult:
    from app.ml import anomaly

    df = df.reset_index(drop=True)
    data = df.drop(columns=[ROW_ID])
    cols = anomaly.numeric_columns(data) if "*" in p.columns else [c for c in p.columns if c in data.columns]
    res = anomaly.detect(data, cols, z_threshold=p.z_threshold)
    bad = sorted(r["row"] for r in res.rows)
    reasons = {r["row"]: "; ".join(f"ANOMALY: {f['column']}={f['value']:g} (median {f['median']:g}, robust z {f['z']})"
                                   for f in r["flags"]) for r in res.rows}
    q = df.loc[bad].copy() if bad else None
    if q is not None:
        q[REASON] = [reasons[i] for i in bad]
    out = df.drop(index=bad)
    r = OpResult(out, quarantined=q)
    r.notes.append(f"Isolation Forest on {', '.join(cols) or 'no numeric columns'}: {len(bad)} rows quarantined for review")
    return r


IMPLEMENTATIONS = {
    "trim_whitespace": trim_whitespace,
    "standardize_case": standardize_case,
    "normalize_email": normalize_email,
    "normalize_phone": normalize_phone,
    "parse_dates": parse_dates,
    "fill_missing": fill_missing,
    "decode_html": decode_html,
    "repair_text": repair_text,
    "redact_text_pii": redact_text_pii,
    "mask_pii": mask_pii,
    "drop_columns": drop_columns,
    "deduplicate": deduplicate,
    "merge_rows": merge_rows,
    "validate_rules": validate_rules,
    "quarantine_anomalies": quarantine_anomalies,
}
assert set(IMPLEMENTATIONS) == set(S.CATALOG), "every catalog operator needs an implementation"
