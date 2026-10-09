"""Virtual Data Tree: React's Virtual DOM idea applied to data hygiene.

The real file is the "real DOM". A dry run mounts an in-memory shadow of it (pandas, all-string,
every row tagged with its original row id), runs the cleaners against the shadow, and reconciles
shadow vs source into a Data-Patch: one entry per deleted, quarantined, redacted or normalized
cell/row. Nothing touches the disk until a human commits the patch (see services/vault.py).

The tree is a pure function of (source bytes, spec), so it can be re-mounted at any time; the
source fingerprint taken at mount time makes a stale patch impossible to commit.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.pipeline import operators as O
from app.pipeline.executor import REDACT_OPS, SENSITIVE, ExecutionResult
from app.scanner import pii
from app.scanner.profiler import date_format

ENGINE = "virtual-data-tree/1.0"
OPS = ("DELETE", "QUARANTINE", "REDACT", "NORMALIZE")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:  # opened read-only: mounting never writes to the source
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _empty(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and v == "")


def _s(v) -> str:
    return "" if _empty(v) else str(v)


# ---------------------------------------------------------------- write-back (in-place candidate)


def in_place_support(path: Path, ext: str, rows: int, max_rows: int, columns: list[str]) -> tuple[bool, str]:
    """Can the shadow be written back over the source without losing anything the shadow never saw?"""
    if rows >= max_rows:
        return False, f"only the first {max_rows:,} rows were mounted; writing back would truncate the file"
    if ext in ("csv", "tsv"):
        return True, ""
    if ext == "json":
        if any("." in c for c in columns):
            return False, "nested JSON was flattened when mounted; write a new file instead"
        return True, ""
    if ext == "xlsx":
        try:
            with pd.ExcelFile(path) as book:
                if len(book.sheet_names) > 1:
                    return False, "the workbook has several sheets; only one was mounted"
        except Exception:  # noqa: BLE001
            return False, "the workbook could not be re-opened"
        return True, ""
    return False, f".{ext} files can only be cleaned into a new file"


def write_like_source(df: pd.DataFrame, path: Path, ext: str) -> None:
    out = df.drop(columns=[c for c in (O.ROW_ID, O.REASON, "__step") if c in df.columns])
    if ext == "csv":
        out.to_csv(path, index=False)
    elif ext == "tsv":
        out.to_csv(path, index=False, sep="\t")
    elif ext == "json":
        recs = [{k: (None if _empty(v) else v) for k, v in r.items()} for r in out.to_dict("records")]
        path.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    elif ext == "xlsx":
        out.to_excel(path, index=False, engine="openpyxl")
    else:
        raise ValueError(f"cannot write .{ext}")


# ---------------------------------------------------------------- the tree


@dataclass
class VirtualTree:
    run_id: str
    asset_name: str
    source: pd.DataFrame                  # mounted shadow of the real file, with __row_id
    result: ExecutionResult               # shadow after the cleaners ran
    sensitive: dict[str, str]             # column -> PII entity (for masking what we display)
    pii_columns: set[str] = field(default_factory=set)  # every column holding personal data (never aggregated)
    patch: list[dict] = field(default_factory=list)
    by_row: dict[int, list[dict]] = field(default_factory=dict)


def reconcile(source: pd.DataFrame, result: ExecutionResult) -> list[dict]:
    """Diff the shadow against the source. Unmasked; callers mask before anything leaves the server."""
    patch: list[dict] = []
    src = source.set_index(O.ROW_ID)
    for rid, (step, kept) in result.removed.items():
        patch.append({"row_id": rid, "op": "DELETE", "step": step, "kept_row": kept,
                      "reason": f"Duplicate of row {kept} (merged into it)"})
    q = result.quarantine
    if len(q):
        for rid, reason, step in zip(q[O.ROW_ID].tolist(), q[O.REASON].tolist(), q["__step"].tolist()):
            patch.append({"row_id": int(rid), "op": "QUARANTINE", "step": step, "reason": str(reason)})
    out = result.output.set_index(O.ROW_ID)
    for (rid, col), (step, op) in result.cell_ops.items():
        if rid not in out.index or col not in out.columns:
            continue  # row later removed or column later dropped: covered by its own entry
        before, after = _s(src.at[rid, col]), _s(out.at[rid, col])
        if before == after:
            continue  # changed and changed back
        patch.append({"row_id": rid, "op": "REDACT" if op in REDACT_OPS else "NORMALIZE", "col": col,
                      "before": before, "after": after, "step": step, "operator": op})
    patch.sort(key=lambda p: (p["row_id"], OPS.index(p["op"]), p.get("col", "")))
    return patch


def build(run_id: str, asset_name: str, source_df: pd.DataFrame, result: ExecutionResult,
          sensitive: dict[str, str]) -> VirtualTree:
    src = source_df.reset_index(drop=True).copy()
    src.insert(0, O.ROW_ID, range(len(src)))
    tree = VirtualTree(run_id, asset_name, src, result, {c: e for c, e in sensitive.items() if e in SENSITIVE},
                       set(sensitive))
    tree.patch = reconcile(src, result)
    for p in tree.patch:
        tree.by_row.setdefault(p["row_id"], []).append(p)
    return tree


def mutations(tree: VirtualTree, source_bytes: int, candidate_bytes: int | None) -> dict:
    n = Counter(p["op"] for p in tree.patch)
    return {
        "pruned_duplicates": n["DELETE"],
        "quarantined_rows": n["QUARANTINE"],
        "masked_pii_cells": n["REDACT"],
        "normalized_cells": n["NORMALIZE"],
        "dropped_columns": sorted(tree.result.dropped_columns),
        "rows_touched": len(tree.by_row),
        "storage_delta_bytes": None if candidate_bytes is None else candidate_bytes - source_bytes,
    }


# ---------------------------------------------------------------- masking for display


def shield(v: str) -> str:
    """Keep only the shape of a value (separators, spacing, length) and hide every character.
    Shows *what changed* (spacing, format) without revealing any part of the identifier; revealing
    even a few characters would add to what the masked display value already shows."""
    return "".join("•" if ch.isalnum() else ch for ch in str(v))


def shield_text(text: str) -> str:
    """Free text: shield only the PII spans so reviewers still see the sentence being redacted."""
    out, last = [], 0
    for _, a, b in sorted(pii._text_hits(text), key=lambda h: h[1]):
        out += [text[last:a], shield(text[a:b])]
        last = b
    return "".join(out + [text[last:]])


def _display(tree: VirtualTree, col: str, v: str) -> str:
    if not v:
        return ""
    return pii.mask_value(tree.sensitive[col], v) if col in tree.sensitive else v


def _display_entry(tree: VirtualTree, p: dict) -> dict:
    e = {k: v for k, v in p.items() if k not in ("before", "after")}
    if "col" in p:
        col, before, after = p["col"], p["before"], p["after"]
        if p["op"] == "REDACT":
            e["before"] = shield(before) if len(before) <= 40 else shield_text(before)
            e["after"] = after
        elif col in tree.sensitive:
            e["before"], e["after"] = shield(before), shield(after)
        else:
            e["before"], e["after"] = before, after
    return e


def masked_patch(tree: VirtualTree, meta: dict) -> dict:
    return {**meta, "patches": [_display_entry(tree, p) for p in tree.patch]}


# ---------------------------------------------------------------- Diff Inspector


def inspector(tree: VirtualTree, op: str | None, offset: int, limit: int) -> dict:
    """Rows in git-diff form: red = deleted/quarantined, yellow = redacted cells, green = normalized cells."""
    rows = sorted(tree.by_row)
    if op:
        rows = [r for r in rows if any(p["op"] == op for p in tree.by_row[r])]
    src = tree.source.set_index(O.ROW_ID)
    columns = [c for c in tree.source.columns if c != O.ROW_ID]
    out = []
    for rid in rows[offset:offset + limit]:
        entries = tree.by_row[rid]
        status = next((p for p in entries if p["op"] in ("DELETE", "QUARANTINE")), None)
        cells = {p["col"]: _display_entry(tree, p) for p in entries if "col" in p}
        values = {c: _display(tree, c, _s(src.at[rid, c])) for c in columns}
        out.append({"row_id": rid, "status": status["op"] if status else "MODIFIED",
                    "reason": status["reason"] if status else "", "kept_row": status.get("kept_row") if status else None,
                    "values": values, "cells": cells})
    counts = Counter(p["op"] for p in tree.patch)
    return {"columns": columns, "dropped_columns": sorted(tree.result.dropped_columns), "rows": out,
            "total_rows": len(rows), "offset": offset, "limit": limit, "counts": {k: counts[k] for k in OPS}}


# ---------------------------------------------------------------- downstream impact simulation


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def impact(tree: VirtualTree, spec_steps: list[dict]) -> dict:
    """What an analytics pipeline reading this file would notice if the patch were committed."""
    from app.ml import anomaly

    src = tree.source.drop(columns=[O.ROW_ID])
    out = tree.result.output.drop(columns=[O.ROW_ID])
    q = tree.result.quarantine
    checks: list[dict] = []

    dropped = sorted(tree.result.dropped_columns)
    checks.append({"name": "Schema contract", "status": "warn" if dropped else "pass",
                   "detail": (f"{len(dropped)} column(s) removed: {', '.join(dropped)}. Downstream queries selecting them will break."
                              if dropped else f"All {len(src.columns)} columns kept with the same names and order.")})
    lost = {c: int(src[c].map(lambda v: not _empty(v) and str(v).strip() != "").sum()) for c in dropped if c in src.columns}
    if any(lost.values()):
        checks.append({"name": "Values lost with dropped columns", "status": "warn",
                       "detail": "; ".join(f"'{c}' still held {n:,} value{'s' if n != 1 else ''}" for c, n in lost.items() if n)
                                 + ". They are not in the quarantine file; keep the column unless these values are junk."})

    n_del = len(tree.result.removed)
    ok = len(out) + len(q) + n_del == len(src)
    checks.append({"name": "Row accounting", "status": "pass" if ok else "fail",
                   "detail": f"{len(out):,} kept + {len(q):,} quarantined + {n_del:,} merged = {len(out) + len(q) + n_del:,} of {len(src):,}. "
                             "Quarantined rows are kept in a review file, not lost."})

    if n_del:  # a column whose every value is unique and present names one record/event per row
        q_ids = set(q[O.ROW_ID].tolist()) if len(q) else set()
        gone = [r for r in tree.result.removed if r not in q_ids]
        for col in src.columns:
            vals = src[col]
            if col in tree.pii_columns or not vals.map(lambda v: not _empty(v)).all() or vals.nunique() != len(vals):
                continue
            ev = bool(re.search(r"ticket|order|txn|transaction|invoice|booking|payment|visit|appointment|claim|event|case", col, re.I))
            checks.append({"name": f"Unique IDs merged away ({col})", "status": "warn" if ev else "info",
                           "detail": f"{len(gone):,} rows with their own {col} were merged into another row. "
                                     + ("Each one looks like a separate event, so check these are true duplicates."
                                        if ev else "Fine if these are copies of one record; not if each ID is a different one.")})
            break

    keys = [k for s in spec_steps if s["op"] == "deduplicate" for k in s["params"].get("keys", [])]
    for k in dict.fromkeys(keys):
        if k not in src.columns or k not in out.columns:
            continue
        norm = lambda vals: {O._key_norm(k, v) for v in vals} - {""}  # noqa: E731
        before = norm(src[k].tolist())
        after = norm(out[k].tolist()) | (norm(q[k].tolist()) if k in q.columns else set())
        lost = len(before - after)
        checks.append({"name": f"Entity preservation ({k})", "status": "pass" if not lost else "warn",
                       "detail": f"{len(before):,} distinct {k} values before, {len(after):,} after (kept + quarantined)."
                                 + ("" if not lost else f" {lost:,} disappeared.")})

    for col, ent in tree.sensitive.items():
        if col not in out.columns:
            continue
        redacted = {rid for (rid, c), (_, op) in tree.result.cell_ops.items() if c == col and op in REDACT_OPS}
        o = tree.result.output
        raw_after = sum(1 for r, v in zip(o[O.ROW_ID].tolist(), o[col].tolist()) if not _empty(v) and r not in redacted)
        raw_before = int(src[col].map(lambda v: not _empty(v)).sum())
        checks.append({"name": f"PII exposure ({col})", "status": "pass" if raw_after == 0 else "info",
                       "detail": f"Readable {ent} values: {raw_before:,} before, {raw_after:,} after."})

    for col in out.columns:
        vals = [str(v).strip() for v in src[col].tolist() if not _empty(v)][:5000]
        if not vals or sum(1 for v in vals if date_format(v)) < 0.6 * len(vals):
            continue
        fb = Counter(date_format(v) for v in vals if date_format(v))
        fa = Counter(date_format(str(v).strip()) for v in out[col].tolist() if not _empty(v) and date_format(str(v).strip()))
        if len(fb) != len(fa):
            checks.append({"name": f"Date format ({col})", "status": "pass" if len(fa) <= 1 else "info",
                           "detail": f"{len(fb)} date formats before ({', '.join(f for f, _ in fb.most_common(3))}), {len(fa)} after."})

    aggregates = []
    for col in [c for c in anomaly.numeric_columns(src) if c not in tree.pii_columns][:4]:
        if col not in out.columns:
            continue
        a, b = _num(src[col]), _num(out[col])
        if a.notna().sum() == 0 or b.notna().sum() == 0:
            continue
        ma, mb, meda, medb = float(a.mean()), float(b.mean()), float(a.median()), float(b.median())
        drift = 0.0 if ma == 0 else (mb - ma) / abs(ma) * 100
        med_drift = 0.0 if meda == 0 else (medb - meda) / abs(meda) * 100
        aggregates.append({"column": col, "sum_before": round(float(a.sum()), 2), "sum_after": round(float(b.sum()), 2),
                           "mean_before": round(ma, 2), "mean_after": round(mb, 2), "mean_drift_pct": round(drift, 2),
                           "median_before": round(meda, 2), "median_after": round(medb, 2), "median_drift_pct": round(med_drift, 2)})
        if abs(drift) > 5:
            why = (f" The median barely moves ({meda:,.0f} → {medb:,.0f}), so the shift comes from a few extreme rows leaving the file."
                   if abs(med_drift) < 2 else "")
            checks.append({"name": f"Aggregate drift ({col})", "status": "warn",
                           "detail": f"Mean moves {drift:+.1f}% ({ma:,.0f} → {mb:,.0f}). Dashboards averaging {col} will shift.{why}"})

    distributions = []
    for col in src.columns:
        if col not in out.columns or col in tree.pii_columns:
            continue
        vc = src[col].dropna().astype(str).str.strip().str.lower().value_counts()
        if not 2 <= len(vc) <= 12 or vc.sum() < 20:
            continue
        va = out[col].dropna().astype(str).str.strip().str.lower().value_counts()
        sb, sa = vc / vc.sum(), (va / va.sum()) if va.sum() else va
        shift = max(abs(sa.get(k, 0) - sb.get(k, 0)) for k in set(sb.index) | set(sa.index)) * 100
        distributions.append({"column": col, "max_share_shift_pp": round(float(shift), 2),
                              "top": [{"value": k, "before": round(float(sb.get(k, 0)) * 100, 1),
                                       "after": round(float(sa.get(k, 0)) * 100, 1)} for k in vc.index[:5]]})
        if len(distributions) >= 3:
            break

    order = {"fail": 0, "warn": 1, "info": 2, "pass": 3}
    checks.sort(key=lambda c: order[c["status"]])
    return {"safe": not any(c["status"] == "fail" for c in checks),
            "checks": checks, "aggregates": aggregates, "distributions": distributions}


# ---------------------------------------------------------------- mount cache

_LOCK = threading.Lock()
_TREES: "OrderedDict[str, VirtualTree]" = OrderedDict()
_MAX = 4


def remember(tree: VirtualTree) -> None:
    with _LOCK:
        _TREES[tree.run_id] = tree
        _TREES.move_to_end(tree.run_id)
        while len(_TREES) > _MAX:
            _TREES.popitem(last=False)


def cached(run_id: str) -> VirtualTree | None:
    with _LOCK:
        t = _TREES.get(run_id)
        if t:
            _TREES.move_to_end(run_id)
        return t


def unmount(run_id: str) -> None:
    with _LOCK:
        _TREES.pop(run_id, None)
