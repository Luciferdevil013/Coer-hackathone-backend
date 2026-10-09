"""Unused-column detection from database query logs.

Usage cannot be inferred from a data dump. We need query history (Postgres
pg_stat_statements, Snowflake ACCESS_HISTORY, BigQuery INFORMATION_SCHEMA.JOBS...). The
connector accepts a CSV export: timestamp, system, query.
"""
from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

from app.engine.signals import ScanContext, Signal, ev


def load_query_log(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            try:
                ts = datetime.fromisoformat(r["timestamp"].replace("Z", ""))
            except (KeyError, ValueError):
                continue
            rows.append({"ts": ts, "system": r.get("system", "unknown"), "query": r.get("query", "")})
    return rows


def _rx(word: str) -> re.Pattern:
    return re.compile(r"(?<![\w])" + re.escape(word.lower()) + r"(?![\w])")


def detect_unused_columns(ctx: ScanContext) -> list[Signal]:
    log = ctx.query_log
    if not log:
        return []
    s = ctx.settings
    first, last = min(r["ts"] for r in log), max(r["ts"] for r in log)
    coverage_days = (last - first).days
    lowered = [(r["ts"], r["system"], r["query"].lower()) for r in log]
    signals = []
    for a in ctx.active():
        schema = a.meta.get("sqlite") or {}
        for table, info in (schema.get("tables") or {}).items():
            trx = _rx(table)
            tq = [(ts, sysname, q) for ts, sysname, q in lowered if trx.search(q)]
            if not tq:
                continue
            wildcard_recent = [1 for ts, _, q in tq if re.search(r"select\s+\*", q) and (ctx.now - ts).days <= s.usage_window_days]
            unused = []
            for col in info["columns"]:
                if col in info.get("pk", []) or col.lower() in ("id", "rowid"):
                    continue
                crx = _rx(col)
                hits = [(ts, sysname) for ts, sysname, q in tq if crx.search(q)]
                last_ts = max((h[0] for h in hits), default=None)
                days = (ctx.now - last_ts).days if last_ts else None
                if last_ts is None or days > s.usage_window_days:
                    recent_systems = {h[1] for h in hits if (ctx.now - h[0]).days <= s.usage_window_days}
                    unused.append({
                        "column": col,
                        "last_referenced_days": days,
                        "last_referenced": last_ts.strftime("%d %b %Y") if last_ts else "never (in log)",
                        "systems_using": len(recent_systems),
                        "historic_systems": sorted({h[1] for h in hits}),
                        "non_null": info["non_null"].get(col, 0),
                        "bytes": info.get("bytes", {}).get(col, 0),
                    })
            if not unused:
                continue
            col_bytes = sum(u["bytes"] for u in unused)
            conf = 0.9 if coverage_days >= s.usage_window_days else 0.6
            if wildcard_recent:
                conf -= 0.15
            names = ", ".join(u["column"] for u in unused)
            months = (min((u["last_referenced_days"] or coverage_days) for u in unused)) // 30
            signals.append(Signal(
                kind="unused_columns", category="stale",
                title=f"{len(unused)} unused column{'s' if len(unused) > 1 else ''} in {a.name} → {table}",
                summary=(f"{names}: not referenced by any detected workflow for {months}+ months. "
                         "Archive the values to cold storage, then drop after DBA approval."),
                target_ids=[a.id],
                evidence=[
                    ev("Table", f"{table} ({info['rows']:,} rows)"),
                    *[ev(u["column"], f"last referenced {u['last_referenced']} · used by {u['systems_using']} systems · "
                                      f"{u['non_null']:,} non-empty values") for u in unused],
                    ev("Query log coverage", f"{len(log):,} queries over {coverage_days} days"),
                    ev("Wildcard (SELECT *) queries in window", len(wildcard_recent)),
                ],
                details={"table": table, "columns": unused, "column_bytes": col_bytes, "coverage_days": coverage_days},
                confidence=round(conf, 3), action="ARCHIVE", records=info["rows"],
            ))
    return signals
