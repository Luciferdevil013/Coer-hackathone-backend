"""Anomalous values detector (Isolation Forest fitted per dataset + robust z-score explanation)."""
from __future__ import annotations

from app.engine.signals import ScanContext, Signal, ev
from app.ml import anomaly

MAX_ROWS_SHOWN = 60


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:,.2f}"


def detect_anomalies(ctx: ScanContext) -> list[Signal]:
    z = ctx.settings.anomaly_z_threshold
    signals = []
    for asset_id, view in ctx.views.items():
        a = ctx.assets[asset_id]
        if a.status != "active" or a.source == "pipeline-output" or asset_id in ctx.doomed:
            continue
        cols = anomaly.numeric_columns(view.df, view.profiles)
        if not cols:
            continue
        res = anomaly.detect(view.df, cols, z_threshold=z)
        if not res.count:
            continue
        by_col: dict[str, int] = {}
        for r in res.rows:
            for f in r["flags"]:
                by_col[f["column"]] = by_col.get(f["column"], 0) + 1
        worst = max((f for r in res.rows for f in r["flags"]), key=lambda f: abs(f["z"]))
        shown = []
        for r in res.rows[:MAX_ROWS_SHOWN]:
            rec = {"row": r["row"], "score": r["score"], "flags": r["flags"]}
            ctx_cols = [c for c in view.df.columns if c not in cols][:3]
            rec["context"] = {c: (str(view.df.iat[r["row"], view.df.columns.get_loc(c)])[:40]
                                  if view.profiles[c].pii_entity is None else "•••") for c in ctx_cols}
            shown.append(rec)
        mean_z = sum(max(abs(f["z"]) for f in r["flags"]) for r in res.rows) / res.count
        conf = round(min(0.97, 0.7 + 0.03 * min(mean_z, 9)), 3)
        cols_txt = ", ".join(f"'{c}'" for c in by_col)
        signals.append(Signal(
            kind="anomalous_values", category="quality",
            title=f"{res.count} anomalous record{'s' if res.count != 1 else ''} in {a.name}",
            summary=(f"Isolation Forest flagged rows that are unusual across {len(cols)} numeric columns; "
                     f"the strongest is {worst['column']} = {_fmt(worst['value'])} (median {_fmt(worst['median'])}, "
                     f"{abs(worst['z'])} robust deviations). Quarantine them for review before they skew reports."),
            target_ids=[a.id],
            evidence=[
                ev("Model", "Isolation Forest (scikit-learn, 200 trees, fitted on this dataset)"),
                ev("Columns analysed", ", ".join(cols)),
                ev("Rows scanned", f"{view.rows:,}"),
                ev("Anomalous rows", f"{res.count:,} ({res.count / max(1, view.rows):.2%})"),
                *[ev(f"Flagged in '{c}'", f"{n:,} values") for c, n in by_col.items()],
                ev("Rule", f"Isolation Forest outlier AND robust z ≥ {res.params['robust_z_threshold']} (adjusted for {view.rows:,} rows), "
                           "or an amount ≥ 100× off its median"),
            ],
            details={"rows": shown, "columns": cols, "by_column": by_col, "dataset_rows": view.rows,
                     "params": res.params,
                     "suggested_prompt": f"Quarantine rows with anomalous values in {cols_txt}"},
            confidence=conf, action="CLEAN", records=res.count,
        ))
    return signals
