"""Scan orchestration: connectors -> content readers -> detectors -> decision engine -> findings."""
from __future__ import annotations

import logging
import time
import traceback

from sqlalchemy import select

from app.config import get_settings
from app.db import session_scope
from app.engine.decision import decide
from app.engine.health import record_snapshot
from app.engine.signals import ScanContext, Signal, ev
from app.models import Asset, AuditEvent, Finding, Scan, utcnow
from app.scanner import anomalies, readers, redundancy, rows, sensitivity, usage
from app.scanner.inventory import sync_inventory, sync_local
from app.scanner.profiler import DatasetView, profile_dataframe
from app.services.storage import Layout, asset_path

log = logging.getLogger("dataguardian.scan")


def _progress(scan_id: str, stage: str, progress: float) -> None:
    with session_scope() as db:
        s = db.get(Scan, scan_id)
        if s:
            s.stage, s.progress, s.status = stage, round(progress, 3), "running"


def load_content(ctx: ScanContext, layout: Layout) -> int:
    s = ctx.settings
    scanned = 0
    for a in ctx.active():
        if not a.content_available or a.size_bytes > s.max_content_bytes:
            continue
        path = asset_path(layout, a)
        if path is None or not path.exists():
            continue
        meta = dict(a.meta or {})
        meta.pop("read_error", None)
        if a.kind == "tabular":
            errs: list[str] = []
            df = readers.read_table(path, s.max_rows_per_dataset, errs)
            if df is not None and len(df.columns):
                v = DatasetView(a.id, a.name, df)
                profile_dataframe(v, s.pii_sample_values, s.pii_column_match_ratio)
                ctx.views[a.id] = v
                meta.update(rows=len(df), columns=len(df.columns), truncated=len(df) >= s.max_rows_per_dataset)
            elif a.ext == "json":  # json that isn't a record list -> treat as text
                ctx.texts[a.id] = readers.read_text(path)
            else:
                meta["read_error"] = errs[0] if errs else "The file has no columns"
        elif a.kind == "binary":
            meta["read_error"] = (f"'.{a.ext}' files are not a supported format, so the contents were not scanned"
                                  if a.ext else "Unknown file type, so the contents were not scanned")
        elif a.kind == "database":
            schema = readers.sqlite_schema(path)
            meta["sqlite"] = schema
            if schema.get("error"):
                meta["read_error"] = f"Could not open the database: {schema['error'][:200]}"
            tviews = []
            for t, info in schema.get("tables", {}).items():
                df = readers.sqlite_table(path, t, s.max_rows_per_dataset)
                if df is not None and len(df):
                    tv = DatasetView(a.id, t, df)
                    profile_dataframe(tv, s.pii_sample_values, s.pii_column_match_ratio)
                    tviews.append(tv)
            ctx.db_views[a.id] = tviews
            meta.update(tables=len(schema.get("tables", {})), rows=sum(t["rows"] for t in schema.get("tables", {}).values()))
        elif a.kind in ("document", "text"):
            ctx.texts[a.id] = readers.read_text(path)
        elif a.kind == "archive" and a.ext == "zip":
            meta["zip"] = readers.zip_listing(path)
        a.meta = meta
        scanned += 1
    return scanned


def unreadable_signals(ctx: ScanContext) -> list[Signal]:
    """Never skip a file silently: anything we could not open is reported, with the reason."""
    bad = [a for a in ctx.active() if (a.meta or {}).get("read_error")]
    if not bad:
        return []
    return [Signal(
        kind="unreadable_files", category="quality",
        title=f"{len(bad)} file{'s' if len(bad) != 1 else ''} could not be scanned",
        summary="Their contents were not checked for duplicates, personal data or quality issues. "
                "Re-save them in a supported format (CSV, XLSX, JSON, PDF, DOCX, TXT) and re-scan.",
        target_ids=[a.id for a in bad],
        evidence=[ev(a.name, a.meta["read_error"]) for a in bad[:12]],
        details={"files": [{"asset_id": a.id, "name": a.name, "reason": a.meta["read_error"]} for a in bad]},
        confidence=1.0, action="KEEP",
    )]


def run_scan(scan_id: str) -> None:
    settings = get_settings()
    t0 = time.perf_counter()
    try:
        with session_scope() as db:
            scan = db.get(Scan, scan_id)
            ws_id = scan.workspace_id
        layout = Layout(ws_id).ensure()

        _progress(scan_id, "Discovering files and connectors", 0.05)
        with session_scope() as db:
            n_local, _ = sync_local(db, ws_id, layout)
            n_inv = sync_inventory(db, ws_id, layout)

        # Detectors run on detached objects so no write transaction is held while progress updates.
        with session_scope() as db:
            assets = {a.id: a for a in db.scalars(select(Asset).where(Asset.workspace_id == ws_id))}
            dismissed = {
                f.details.get("fingerprint"): f.resolution
                for f in db.scalars(select(Finding).where(Finding.workspace_id == ws_id, Finding.status == "dismissed"))
            }
            db.expunge_all()
        ctx = ScanContext(workspace_id=ws_id, settings=settings, now=utcnow(), assets=assets)
        qlog = layout.connectors / "query_log.csv"
        ctx.query_log = usage.load_query_log(qlog) if qlog.exists() else None

        _progress(scan_id, "Reading and profiling content", 0.15)
        scanned = load_content(ctx, layout)

        signals: list[Signal] = []
        _progress(scan_id, "Detecting sensitive data (Aadhaar, PAN, cards, secrets)", 0.45)
        signals += sensitivity.detect_sensitive(ctx)
        signals += sensitivity.detect_text_pii(ctx)
        _progress(scan_id, "Finding redundant backups and versions", 0.6)
        signals += redundancy.detect_backups(ctx)
        signals += redundancy.detect_exact_duplicates(ctx)
        signals += redundancy.detect_similar_clusters(ctx)
        _progress(scan_id, "Checking staleness and column usage", 0.72)
        signals += sensitivity.detect_stale_large(ctx)
        signals += usage.detect_unused_columns(ctx)
        # don't ask users to clean files we already recommend deleting
        ctx.doomed = {p["asset_id"] for sig in signals for p in (sig.plan or []) if p["action"] == "DELETE"}
        ctx.doomed |= {t for sig in signals if sig.action == "DELETE" and not sig.plan for t in sig.target_ids}
        _progress(scan_id, "Matching duplicate records and data quality", 0.82)
        signals += rows.detect_duplicate_rows(ctx)
        signals += rows.detect_quality(ctx)
        _progress(scan_id, "Fitting Isolation Forest for anomalous values", 0.88)
        signals += anomalies.detect_anomalies(ctx)
        signals += unreadable_signals(ctx)
        _progress(scan_id, "Scoring recommendations", 0.94)

        with session_scope() as db:
            for a in assets.values():
                row = db.get(Asset, a.id)
                if row is not None and row.meta != a.meta:
                    row.meta = a.meta
            for sig in signals:
                vals = decide(sig, assets, settings)
                f = Finding(workspace_id=ws_id, scan_id=scan_id, **vals)
                if vals["details"]["fingerprint"] in dismissed:
                    f.status, f.resolution = "dismissed", dismissed[vals["details"]["fingerprint"]]
                db.add(f)

            scan = db.get(Scan, scan_id)
            scan.status, scan.stage, scan.progress = "completed", "Completed", 1.0
            scan.finished_at = utcnow()
            scan.stats = {
                "assets": len([a for a in assets.values() if a.status == "active"]),
                "local_files": n_local, "inventory_objects": n_inv, "content_scanned": scanned,
                "datasets_profiled": len(ctx.views) + sum(len(v) for v in ctx.db_views.values()),
                "unreadable": [a.name for a in assets.values() if a.status == "active" and (a.meta or {}).get("read_error")],
                "findings": len(signals), "duration_s": round(time.perf_counter() - t0, 2),
            }
            db.add(AuditEvent(workspace_id=ws_id, actor="system", event="scan.completed", resource_type="scan",
                              resource_id=scan_id, data=scan.stats))
            record_snapshot(db, ws_id, settings, "Scan completed")
    except Exception as e:  # surface failure to the UI instead of hanging
        log.exception("scan failed")
        with session_scope() as db:
            scan = db.get(Scan, scan_id)
            if scan:
                scan.status, scan.stage, scan.error = "failed", "Failed", f"{e}\n{traceback.format_exc()[-1500:]}"
                scan.finished_at = utcnow()
