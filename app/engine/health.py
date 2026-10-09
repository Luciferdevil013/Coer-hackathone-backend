"""Dashboard metrics and the transparent Data Health Score."""
from __future__ import annotations

from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Asset, Finding, HealthSnapshot, Scan, utcnow

FORMULA = (
    "100 − 35×(redundant bytes ÷ total) − min(35, Σ open risk scores × 0.06) "
    "− 20×(stale bytes ÷ total) − min(10, Σ quality issue rates × 20)"
)


def latest_scan(db: Session, workspace_id: str) -> Scan | None:
    return db.scalars(
        select(Scan).where(Scan.workspace_id == workspace_id, Scan.status == "completed").order_by(Scan.finished_at.desc())
    ).first()


def compute_summary(db: Session, workspace_id: str, s: Settings, include_extras: bool = True) -> dict:
    scan = latest_scan(db, workspace_id)
    assets = {a.id: a for a in db.scalars(select(Asset).where(Asset.workspace_id == workspace_id))}
    active = {k: a for k, a in assets.items() if a.status == "active"}
    total = sum(a.size_bytes for a in active.values())
    findings = []
    if scan:
        findings = list(db.scalars(select(Finding).where(Finding.workspace_id == workspace_id, Finding.scan_id == scan.id)))
    open_f = [f for f in findings if f.status == "open"]

    def union_bytes(fs, actions=("DELETE", "ARCHIVE")) -> tuple[int, set[str]]:
        ids: set[str] = set()
        for f in fs:
            for p in f.details.get("plan", []):
                if p["action"] in actions and p["asset_id"] in active:
                    ids.add(p["asset_id"])
        return sum(active[i].size_bytes for i in ids), ids

    redundant_b, redundant_ids = union_bytes([f for f in open_f if f.category == "redundant"])
    stale_b, stale_ids = union_bytes([f for f in open_f if f.category == "stale" and f.kind != "unused_columns"])
    stale_b += sum(f.bytes_affected for f in open_f if f.kind == "unused_columns")
    risk_f = [f for f in open_f if f.category == "risk"]
    risky_ids = {t for f in risk_f for t in f.target_asset_ids if t in active}
    risky_b = sum(active[i].size_bytes for i in risky_ids)
    risky_records = sum(f.records_affected for f in risk_f if f.kind == "sensitive_data")
    quality_f = [f for f in open_f if f.category == "quality"]

    pen_red = 35 * (redundant_b / total) if total else 0
    pen_risk = min(35.0, sum(f.risk_score for f in risk_f) * 0.06)
    pen_stale = 20 * (min(stale_b, total) / total) if total else 0
    q_rates = [f.details.get("issue_rate", 0) for f in quality_f if f.kind == "data_quality"]
    q_rates += [f.records_affected / max(1, f.details.get("dataset_rows", 1)) for f in quality_f if f.kind in ("duplicate_rows", "anomalous_values")]
    pen_q = min(10.0, sum(q_rates) * 20)
    score = max(0, round(100 - pen_red - pen_risk - pen_stale - pen_q)) if scan else None

    flagged = redundant_ids | stale_ids | risky_ids | {t for f in quality_f for t in f.target_asset_ids}
    sev = Counter(f.severity for f in open_f)
    by_kind = Counter()
    bytes_by_kind = Counter()
    for f in open_f:
        by_kind[f.kind] += 1
        bytes_by_kind[f.kind] += f.bytes_affected

    dup_files = sum(
        sum(1 for p in f.details.get("plan", []) if p["action"] == "DELETE")
        for f in open_f if f.kind in ("exact_duplicate", "near_duplicate", "version_chain")
    )
    waste = {
        "redundant": bytes_by_kind["exact_duplicate"] + bytes_by_kind["near_duplicate"] + bytes_by_kind["version_chain"],
        "backups": bytes_by_kind["redundant_backup"],
        "stale": bytes_by_kind["stale_large"],
        "unused_columns": bytes_by_kind["unused_columns"],
    }
    resolved = [f for f in findings if f.status == "resolved"]
    extras = _extras(db, workspace_id, active, open_f, redundant_ids | stale_ids, risky_ids) if include_extras else {}
    return {
        **extras,
        "has_scan": scan is not None,
        "scan": {"id": scan.id, "finished_at": scan.finished_at.isoformat() if scan.finished_at else None,
                 "stats": scan.stats} if scan else None,
        "health_score": score,
        "score_breakdown": {
            "redundancy": round(pen_red, 1), "risk": round(pen_risk, 1),
            "staleness": round(pen_stale, 1), "quality": round(pen_q, 1), "formula": FORMULA,
        },
        "totals": {
            "assets": len(active),
            "total_bytes": total,
            "useful_bytes": max(0, total - redundant_b - min(stale_b, total - redundant_b)),
            "redundant_bytes": redundant_b,
            "stale_bytes": stale_b,
            "risky_bytes": risky_b,
            "risky_records": risky_records,
            "archived_assets": sum(1 for a in assets.values() if a.status == "archived"),
            "trashed_assets": sum(1 for a in assets.values() if a.status == "trashed"),
        },
        "severity": {"critical": sev["critical"], "warning": sev["warning"], "info": sev["info"]},
        "healthy_pct": round(100 * (1 - len(flagged) / len(active)), 1) if active else 100.0,
        "recommendations": {
            "duplicate_bytes": waste["redundant"],
            "backup_bytes": waste["backups"],
            "duplicate_files": dup_files,
            "exposed_sensitive": sum(1 for f in risk_f if f.kind == "sensitive_data" and f.severity == "critical"),
            "secrets": by_kind["secret_exposed"],
            "duplicate_records": sum(f.records_affected for f in open_f if f.kind == "duplicate_rows"),
            "annual_waste_inr": round(sum(f.savings_inr_year for f in open_f), 0),
        },
        "waste_breakdown": waste,
        "resolved": {
            "count": len(resolved),
            "savings_inr_year": round(sum(f.savings_inr_year for f in resolved if f.resolution in ("DELETE", "ARCHIVE")), 0),
        },
        "cost_model": {
            "hot_inr_per_gb_year": round(s.inr_hot_per_gb_year, 2),
            "archive_inr_per_gb_year": round(s.inr_archive_per_gb_year, 3),
            "basis": f"S3 Standard ${s.hot_usd_per_gb_month}/GB-mo vs Glacier Deep Archive ${s.archive_usd_per_gb_month}/GB-mo at ₹{s.usd_to_inr}/USD",
        },
    }


def record_snapshot(db: Session, workspace_id: str, s: Settings, reason: str) -> None:
    db.flush()
    score = compute_summary(db, workspace_id, s, include_extras=False)["health_score"]
    if score is not None:
        db.add(HealthSnapshot(workspace_id=workspace_id, score=score, reason=reason[:200], created_at=utcnow()))


SOURCE_LABEL = {"upload": "Uploaded files", "s3-inventory": "AWS S3 (inventory)", "pipeline-output": "Cleaned outputs"}


def _extras(db: Session, workspace_id: str, active: dict, open_f: list, reclaim_ids: set, risky_ids: set) -> dict:
    flagged_by_asset: dict[str, int] = {}
    for f in open_f:
        for t in f.target_asset_ids:
            flagged_by_asset[t] = flagged_by_asset.get(t, 0) + 1
    by_dept: dict[str, dict] = {}
    by_source: dict[str, dict] = {}
    for a in active.values():
        dept = a.department or "Unassigned"
        d = by_dept.setdefault(dept, {"name": dept, "bytes": 0, "reclaimable": 0})
        d["bytes"] += a.size_bytes
        src = by_source.setdefault(a.source, {"id": a.source, "name": SOURCE_LABEL.get(a.source, a.source), "bytes": 0,
                                              "objects": 0, "issues": 0, "reclaimable": 0, "flagged_bytes": 0})
        src["bytes"] += a.size_bytes
        src["objects"] += 1
        src["issues"] += flagged_by_asset.get(a.id, 0)
        if a.id in reclaim_ids:
            d["reclaimable"] += a.size_bytes
            src["reclaimable"] += a.size_bytes
        if a.id in reclaim_ids or a.id in risky_ids:
            src["flagged_bytes"] += a.size_bytes
    for src in by_source.values():
        src["health"] = round(100 * (1 - src.pop("flagged_bytes") / src["bytes"])) if src["bytes"] else 100
    history = [
        {"at": h.created_at.isoformat(), "score": h.score, "reason": h.reason}
        for h in db.scalars(select(HealthSnapshot).where(HealthSnapshot.workspace_id == workspace_id)
                            .order_by(HealthSnapshot.created_at))
    ]
    return {
        "by_department": sorted(by_dept.values(), key=lambda d: -d["bytes"])[:8],
        "by_source": sorted(by_source.values(), key=lambda d: -d["bytes"]),
        "history": history[-60:],
    }
