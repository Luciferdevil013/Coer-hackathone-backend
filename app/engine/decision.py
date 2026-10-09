"""Data Decision Engine: converts detector Signals into scored, explainable Findings.

All numbers here are deterministic. The LLM never sets an action, a confidence or a saving.
"""
from __future__ import annotations

import hashlib
import math

from app.config import Settings
from app.engine.signals import Signal
from app.models import Asset

GB = 1024**3
SEVERITY_RANK = {"critical": 3, "warning": 2, "info": 1}
DEFAULT_ACTION = {
    "exact_duplicate": "DELETE", "version_chain": "ARCHIVE", "near_duplicate": "ARCHIVE",
    "redundant_backup": "DELETE", "stale_large": "ARCHIVE", "sensitive_data": "SECURE",
    "secret_exposed": "SECURE", "duplicate_rows": "CLEAN", "data_quality": "CLEAN", "unused_columns": "ARCHIVE",
    "anomalous_values": "CLEAN", "pii_in_text": "CLEAN", "unreadable_files": "KEEP",
}


def hot_cost_year(nbytes: int, s: Settings) -> float:
    return nbytes / GB * s.inr_hot_per_gb_year


def saving_for(action: str, asset: Asset, s: Settings, nbytes: int | None = None) -> float:
    b = asset.size_bytes if nbytes is None else nbytes
    if action == "DELETE":
        return b / GB * (s.inr_hot_per_gb_year if asset.storage_class == "STANDARD" else s.inr_archive_per_gb_year)
    if action == "ARCHIVE" and asset.storage_class == "STANDARD":
        return b / GB * (s.inr_hot_per_gb_year - s.inr_archive_per_gb_year)
    return 0.0


def fingerprint(sig: Signal) -> str:
    key = sig.kind + "|" + ",".join(sorted(sig.target_ids)) + "|" + str(sig.details.get("table", ""))
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _severity(sig: Signal, nbytes: int, savings: float, rate: float | None) -> str:
    if sig.category == "risk":
        if sig.risk >= 70:
            return "critical"
        return "warning" if sig.risk >= 40 else "info"
    if sig.category == "quality":
        if sig.kind == "anomalous_values":
            extreme = any(abs(f["z"]) >= 10 for r in sig.details.get("rows", []) for f in r["flags"])
            return "warning" if extreme or sig.records >= 10 else "info"
        if sig.kind == "duplicate_rows":
            conflicts = any(c.get("conflict_columns") for c in sig.details.get("clusters", []))
            share = sig.records / max(1, sig.details.get("dataset_rows", 1))
            return "warning" if share >= 0.01 or conflicts else "info"
        return "warning" if (rate or 0) >= 0.02 else "info"
    if nbytes >= 100 * GB or savings >= 10_000:
        return "critical"
    if nbytes >= 1 * GB or savings >= 500 or len(sig.target_ids) >= 3:
        return "warning"
    return "info"


def decide(sig: Signal, assets: dict[str, Asset], s: Settings) -> dict:
    """Return the Finding column values for one Signal."""
    action = sig.action or DEFAULT_ACTION.get(sig.kind, "KEEP")
    plan = sig.plan or [{"asset_id": t, "action": action, "reason": ""} for t in sig.target_ids]
    nbytes = 0
    savings = 0.0
    if sig.kind == "unused_columns":
        nbytes = int(sig.details.get("column_bytes", 0))
        a = assets[sig.target_ids[0]]
        savings = saving_for("ARCHIVE", a, s, nbytes)
    else:
        for p in plan:
            if p["action"] in ("DELETE", "ARCHIVE"):
                a = assets[p["asset_id"]]
                nbytes += a.size_bytes
                savings += saving_for(p["action"], a, s)
        if sig.category in ("risk", "quality"):
            nbytes = sum(assets[t].size_bytes for t in sig.target_ids)
    rate = sig.details.get("issue_rate")
    severity = _severity(sig, nbytes, savings, rate)
    priority = (
        SEVERITY_RANK[severity] * 1000
        + sig.risk * 5
        + math.log10(savings + 1) * 60
        + sig.confidence * 20
    )
    return {
        "kind": sig.kind,
        "category": sig.category,
        "title": sig.title,
        "summary": sig.summary,
        "severity": severity,
        "action": action,
        "confidence": round(sig.confidence, 3),
        "bytes_affected": int(nbytes),
        "records_affected": int(sig.records),
        "savings_inr_year": round(savings, 2),
        "risk_score": sig.risk,
        "priority": round(priority, 2),
        "target_asset_ids": sig.target_ids,
        "keep_asset_ids": sig.keep_ids,
        "evidence": sig.evidence,
        "details": {**sig.details, "plan": plan, "fingerprint": fingerprint(sig)},
    }
