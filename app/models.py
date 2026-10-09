"""ORM models. Every tenant-owned row carries workspace_id; all queries filter on it."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Workspace(Base):
    __tablename__ = "workspaces"
    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(120))
    industry: Mapped[str] = mapped_column(String(60), default="")
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class RuleSet(Base):
    """Versioned tenant rules (FR-07). A run pins the version it used."""
    __tablename__ = "rule_sets"
    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=new_id)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    rules: Mapped[dict] = mapped_column(JSON)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_by: Mapped[str] = mapped_column(String(60), default="system")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Asset(Base):
    __tablename__ = "assets"
    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=new_id)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    source: Mapped[str] = mapped_column(String(30))            # upload | s3-inventory | pipeline-output
    path: Mapped[str] = mapped_column(String(600))             # relative path or object key
    name: Mapped[str] = mapped_column(String(300))
    ext: Mapped[str] = mapped_column(String(20), default="")
    kind: Mapped[str] = mapped_column(String(20))              # tabular | database | document | text | archive | binary
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    modified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    accessed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    owner: Mapped[str] = mapped_column(String(120), default="unknown")
    department: Mapped[str] = mapped_column(String(60), default="")
    shared_with: Mapped[int] = mapped_column(Integer, default=1)
    encrypted: Mapped[bool] = mapped_column(Boolean, default=False)
    content_hash: Mapped[str] = mapped_column(String(64), default="", index=True)
    content_available: Mapped[bool] = mapped_column(Boolean, default=True)
    storage_class: Mapped[str] = mapped_column(String(30), default="STANDARD")
    status: Mapped[str] = mapped_column(String(20), default="active")  # active | archived | trashed
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Scan(Base):
    __tablename__ = "scans"
    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=new_id)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    stage: Mapped[str] = mapped_column(String(80), default="Queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stats: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Finding(Base):
    __tablename__ = "findings"
    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=new_id)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    scan_id: Mapped[str] = mapped_column(ForeignKey("scans.id"), index=True)
    kind: Mapped[str] = mapped_column(String(40))
    category: Mapped[str] = mapped_column(String(20))          # redundant | risk | stale | quality
    title: Mapped[str] = mapped_column(String(300))
    summary: Mapped[str] = mapped_column(Text, default="")
    severity: Mapped[str] = mapped_column(String(10))          # critical | warning | info
    action: Mapped[str] = mapped_column(String(10))            # KEEP | CLEAN | ARCHIVE | SECURE | DELETE
    confidence: Mapped[float] = mapped_column(Float)
    bytes_affected: Mapped[int] = mapped_column(Integer, default=0)
    records_affected: Mapped[int] = mapped_column(Integer, default=0)
    savings_inr_year: Mapped[float] = mapped_column(Float, default=0.0)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0)
    priority: Mapped[float] = mapped_column(Float, default=0.0)
    target_asset_ids: Mapped[list] = mapped_column(JSON, default=list)
    keep_asset_ids: Mapped[list] = mapped_column(JSON, default=list)
    evidence: Mapped[list] = mapped_column(JSON, default=list)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    explanation: Mapped[str] = mapped_column(Text, default="")
    explanation_source: Mapped[str] = mapped_column(String(20), default="")
    status: Mapped[str] = mapped_column(String(20), default="open")  # open | resolved | dismissed
    resolution: Mapped[str] = mapped_column(String(20), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Action(Base):
    """An applied, reversible change. `affected` stores enough to undo it."""
    __tablename__ = "actions"
    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=new_id)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    finding_id: Mapped[str | None] = mapped_column(ForeignKey("findings.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(10))
    actor: Mapped[str] = mapped_column(String(60))
    status: Mapped[str] = mapped_column(String(20), default="applied")  # applied | undone
    affected: Mapped[list] = mapped_column(JSON, default=list)
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    undone_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=new_id)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    actor: Mapped[str] = mapped_column(String(60))
    event: Mapped[str] = mapped_column(String(60))
    resource_type: Mapped[str] = mapped_column(String(30))
    resource_id: Mapped[str] = mapped_column(String(40))
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=new_id)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id"))
    prompt: Mapped[str] = mapped_column(Text, default="")
    spec: Mapped[dict] = mapped_column(JSON)
    planner: Mapped[str] = mapped_column(String(20), default="rules")   # llm | rules | manual
    rule_set_version: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="dry_run")  # dry_run | committed | discarded | rolled_back
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    preview: Mapped[dict] = mapped_column(JSON, default=dict)
    staging_path: Mapped[str] = mapped_column(String(600), default="")
    quarantine_path: Mapped[str] = mapped_column(String(600), default="")
    output_asset_id: Mapped[str | None] = mapped_column(String(12), nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")
    approved_by: Mapped[str] = mapped_column(String(60), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    committed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class VaultCommit(Base):
    """An in-place commit of a Data-Patch. The original bytes sit compressed in the vault until rolled back."""
    __tablename__ = "vault_commits"
    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=new_id)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    run_id: Mapped[str] = mapped_column(String(12), index=True)
    asset_id: Mapped[str] = mapped_column(String(12), index=True)
    file_name: Mapped[str] = mapped_column(String(300))
    vault_file: Mapped[str] = mapped_column(String(300))
    codec: Mapped[str] = mapped_column(String(10))                    # zstd | gzip
    original_sha256: Mapped[str] = mapped_column(String(64))
    original_bytes: Mapped[int] = mapped_column(Integer)
    vault_bytes: Mapped[int] = mapped_column(Integer)
    new_sha256: Mapped[str] = mapped_column(String(64))
    new_bytes: Mapped[int] = mapped_column(Integer)
    journal: Mapped[dict] = mapped_column(JSON, default=dict)         # asset state before, quarantine file, resolved findings
    status: Mapped[str] = mapped_column(String(20), default="committed")  # committed | rolled_back
    actor: Mapped[str] = mapped_column(String(60))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    rolled_back_by: Mapped[str] = mapped_column(String(60), default="")
    rolled_back_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class LLMCall(Base):
    """AI observability (M10): every model call is recorded with model, latency and tokens."""
    __tablename__ = "llm_calls"
    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=new_id)
    workspace_id: Mapped[str | None] = mapped_column(String(12), nullable=True)
    purpose: Mapped[str] = mapped_column(String(40))
    model: Mapped[str] = mapped_column(String(60))
    ok: Mapped[bool] = mapped_column(Boolean)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class HealthSnapshot(Base):
    """Score after every state change (scan, action, undo, commit) - drives the trend chart."""
    __tablename__ = "health_snapshots"
    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=new_id)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    score: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
