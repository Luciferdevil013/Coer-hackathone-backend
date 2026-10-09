"""Detector output contract. Detectors emit Signals; the decision engine turns them into Findings."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.config import Settings
from app.models import Asset
from app.scanner.profiler import DatasetView


@dataclass
class Signal:
    kind: str                       # detector id, e.g. "version_chain"
    category: str                   # redundant | risk | stale | quality
    title: str
    summary: str
    target_ids: list[str]           # assets the recommendation acts on
    keep_ids: list[str] = field(default_factory=list)
    plan: list[dict] = field(default_factory=list)   # [{asset_id, action, reason}] per-asset actions
    evidence: list[dict] = field(default_factory=list)  # [{label, value}] - facts only, no opinions
    details: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.9
    action: str | None = None       # detector's proposal; the decision engine has the final say
    records: int = 0
    risk: float = 0.0               # 0..100


def ev(label: str, value: Any) -> dict:
    return {"label": label, "value": value}


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1024 or unit == "PB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


@dataclass
class ScanContext:
    workspace_id: str
    settings: Settings
    now: datetime
    assets: dict[str, Asset]
    views: dict[str, DatasetView] = field(default_factory=dict)       # tabular assets
    db_views: dict[str, list[DatasetView]] = field(default_factory=dict)  # sqlite tables per asset
    texts: dict[str, str] = field(default_factory=dict)
    claimed: set[str] = field(default_factory=set)
    doomed: set[str] = field(default_factory=set)   # assets already recommended for DELETE
    sim_cache: dict = field(default_factory=dict)
    doc_vectors: dict = field(default_factory=dict)
    query_log: list[dict] | None = None

    def active(self, include_outputs: bool = True) -> list[Asset]:
        """Active assets. Pipeline outputs are fresh, lineage-tracked data: excluded from
        redundancy/quality checks but still checked for sensitive data."""
        return [a for a in self.assets.values()
                if a.status == "active" and (include_outputs or a.source != "pipeline-output")]

    def days_since_use(self, a: Asset) -> int:
        ts = a.accessed_at or a.modified_at
        return (self.now - ts).days if ts else 0
