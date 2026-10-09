"""Execute a validated PipelineSpec against a dataset. Pure function of (data, spec)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.pipeline import operators as O
from app.pipeline.spec import CATALOG, PipelineSpec
from app.scanner import pii

SENSITIVE = {"AADHAAR", "PAN", "CREDIT_CARD", "BANK_ACCOUNT", "PASSWORD"}
REDACT_OPS = {"mask_pii", "redact_text_pii"}


@dataclass
class ExecutionResult:
    output: pd.DataFrame
    quarantine: pd.DataFrame
    steps: list[dict] = field(default_factory=list)
    totals: dict = field(default_factory=dict)
    changes: list[dict] = field(default_factory=list)
    # exact provenance for the Data-Patch: which step last wrote each cell (redaction wins),
    # which rows were merged into which survivor, and which step dropped which column
    cell_ops: dict[tuple[int, str], tuple[str, str]] = field(default_factory=dict)
    removed: dict[int, tuple[str, int]] = field(default_factory=dict)
    dropped_columns: dict[str, str] = field(default_factory=dict)


def changed_cells(prev: pd.DataFrame, cur: pd.DataFrame) -> list[tuple[int, str]]:
    """(row id, column) of every cell whose value differs between two versions of the frame.
    Rows missing from `cur` are not cell changes; empty-vs-blank is not a change (as in _map_cells)."""
    cols = [c for c in cur.columns if c != O.ROW_ID and c in prev.columns]
    if not cols or not len(cur):
        return []
    c = cur.set_index(O.ROW_ID)[cols]
    p = prev.set_index(O.ROW_ID)[cols].reindex(c.index)
    pv, cv = p.fillna("").astype(str), c.fillna("").astype(str)
    diff = (pv.to_numpy() != cv.to_numpy())
    blank = (pv.apply(lambda s: s.str.strip() == "").to_numpy() & cv.apply(lambda s: s.str.strip() == "").to_numpy())
    rr, kk = np.nonzero(diff & ~blank)
    ids = c.index.to_numpy()
    return [(int(ids[r]), cols[k]) for r, k in zip(rr, kk)]


def execute(df_in: pd.DataFrame, spec: PipelineSpec) -> ExecutionResult:
    df = df_in.copy().reset_index(drop=True)
    df.insert(0, O.ROW_ID, range(len(df)))
    quarantine_parts: list[pd.DataFrame] = []
    steps, all_changes = [], []
    cell_ops: dict[tuple[int, str], tuple[str, str]] = {}
    removed: dict[int, tuple[str, int]] = {}
    dropped: dict[str, str] = {}
    for step in spec.steps:
        info = CATALOG[step.op]
        params = info.params_model.model_validate(step.params)
        t0 = time.perf_counter()
        rows_in = len(df)
        res = O.IMPLEMENTATIONS[step.op](df, params, None)
        for key in changed_cells(df, res.df):
            if step.op in REDACT_OPS or cell_ops.get(key, ("", ""))[1] not in REDACT_OPS:
                cell_ops[key] = (step.id, step.op)
        removed.update({r: (step.id, kept) for r, kept in res.removed.items()})
        dropped.update({c: step.id for c in df.columns if c not in res.df.columns})
        df = res.df
        q = res.quarantined
        if q is not None and len(q):
            q = q.copy()
            q["__step"] = step.id
            quarantine_parts.append(q)
        steps.append({
            "id": step.id, "op": step.op, "title": info.title,
            "rows_in": rows_in, "rows_out": len(df),
            "cells_changed": res.cells_changed, "rows_removed": res.rows_removed,
            "rows_quarantined": 0 if q is None else len(q),
            "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
            "notes": res.notes,
        })
        for c in res.changes:
            all_changes.append({**c, "step": step.id, "op": step.op})
    quarantine = pd.concat(quarantine_parts, ignore_index=True) if quarantine_parts else pd.DataFrame(columns=[O.ROW_ID, O.REASON])
    totals = {
        "rows_in": len(df_in),
        "rows_out": len(df),
        "cells_changed": sum(s["cells_changed"] for s in steps),
        "duplicates_removed": sum(s["rows_removed"] for s in steps),
        "rows_quarantined": len(quarantine),
        "columns_out": len([c for c in df.columns if c != O.ROW_ID]),
    }
    return ExecutionResult(df, quarantine, steps, totals, all_changes, cell_ops, removed, dropped)


def _mask_frame(df: pd.DataFrame, sensitive_cols: dict[str, str], limit: int) -> list[dict]:
    rows = []
    for _, r in df.head(limit).iterrows():
        rec = {}
        for c, v in r.items():
            if c in (O.ROW_ID,):
                rec["_row"] = int(v)
                continue
            if c == "__step":
                continue
            if v is None or (isinstance(v, float) and pd.isna(v)):
                rec[c] = ""
            elif c in sensitive_cols:
                rec[c] = pii.mask_value(sensitive_cols[c], str(v))
            else:
                rec[c] = str(v)
        rows.append(rec)
    return rows


def preview(result: ExecutionResult, sensitive_cols: dict[str, str], limit: int = 25) -> dict:
    sens = {c: e for c, e in sensitive_cols.items() if e in SENSITIVE}
    per_step: dict[str, int] = {}
    sampled = []
    for ch in result.changes:  # a few examples from every step, not just the first one
        if per_step.get(ch["step"], 0) < 12:
            per_step[ch["step"]] = per_step.get(ch["step"], 0) + 1
            sampled.append(ch)
    changes = []
    for ch in sampled:
        c = dict(ch)
        if c.get("column") in sens:
            c["before"] = pii.mask_value(sens[c["column"]], str(c["before"]))
            c["after"] = pii.mask_value(sens[c["column"]], str(c["after"]))
        changes.append(c)
    return {
        "changes": changes,
        "quarantine": _mask_frame(result.quarantine, sens, limit),
        "output_head": _mask_frame(result.output, sens, limit),
        "columns": [c for c in result.output.columns if c != O.ROW_ID],
    }


def to_csv(df: pd.DataFrame, path, keep_reason: bool = False) -> None:
    if keep_reason:  # quarantine keeps traceability back to the source row and failing step
        out = df.rename(columns={O.ROW_ID: "source_row", O.REASON: "quarantine_reason", "__step": "failed_step"})
    else:
        out = df.drop(columns=[c for c in (O.ROW_ID, "__step", O.REASON) if c in df.columns])
    out.to_csv(path, index=False)
