"""HTTP API. Every resource is addressed under its workspace and re-checked for ownership,
so a valid id from another tenant returns 404 (FR-11)."""
from __future__ import annotations

import shutil
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Body, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.engine.health import compute_summary, latest_scan
from app.llm.explain import explain
from app.llm.gateway import get_gateway
from app.models import Action, Asset, AuditEvent, Finding, LLMCall, PipelineRun, RuleSet, Scan, VaultCommit, Workspace
from app.pipeline.spec import CATALOG
from app.scanner.orchestrator import run_scan
from app.services import pipelines as P
from app.services import vault as VAULT
from app.services import workspaces as W
from app.services.actions import ActionError, apply_action, audit, restore_asset, trash_assets, undo_action
from app.services.storage import Layout, safe_extract_zip, safe_name

router = APIRouter(prefix="/api")


def actor_of(x_user: str | None = Header(default=None)) -> str:
    return (x_user or "admin@demo").strip()[:60]


def get_ws(ws_id: str, db: Session = Depends(get_db)) -> Workspace:
    ws = db.get(Workspace, ws_id)
    if not ws:
        raise HTTPException(404, "workspace not found")
    return ws


def owned(db: Session, model, obj_id: str, ws: Workspace):
    obj = db.get(model, obj_id)
    if obj is None or obj.workspace_id != ws.id:
        raise HTTPException(404, f"{model.__name__.lower()} not found")
    return obj


def bad_request(e: Exception):
    raise HTTPException(400, str(e))


# ------------------------------------------------------------------ serializers


def ser_asset(a: Asset) -> dict:
    return {
        "id": a.id, "source": a.source, "path": a.path, "name": a.name, "ext": a.ext, "kind": a.kind,
        "size_bytes": a.size_bytes, "modified_at": a.modified_at.isoformat() if a.modified_at else None,
        "accessed_at": a.accessed_at.isoformat() if a.accessed_at else None, "owner": a.owner,
        "department": a.department, "shared_with": a.shared_with, "encrypted": a.encrypted,
        "storage_class": a.storage_class, "status": a.status, "content_available": a.content_available,
        "location": (a.meta or {}).get("location", ""),
        "meta": {k: v for k, v in (a.meta or {}).items() if k in ("rows", "columns", "tables", "lineage", "zip", "truncated", "quarantine_file")},
    }


def applied_actions(db: Session, ws_id: str) -> dict[str, str]:
    """finding_id -> id of its latest still-applied action (for inline Undo)."""
    out: dict[str, str] = {}
    for a in db.scalars(select(Action).where(Action.workspace_id == ws_id, Action.status == "applied")
                        .order_by(Action.created_at)):
        if a.finding_id:
            out[a.finding_id] = a.id
    return out


def ser_finding(f: Finding, full: bool = False, action_id: str | None = None) -> dict:
    d = {
        "id": f.id, "kind": f.kind, "category": f.category, "title": f.title, "summary": f.summary,
        "severity": f.severity, "action": f.action, "confidence": f.confidence, "bytes_affected": f.bytes_affected,
        "records_affected": f.records_affected, "savings_inr_year": f.savings_inr_year, "risk_score": f.risk_score,
        "priority": f.priority, "status": f.status, "resolution": f.resolution,
        "target_count": len(f.target_asset_ids), "target_asset_ids": f.target_asset_ids,
        "keep_asset_ids": f.keep_asset_ids, "created_at": f.created_at.isoformat(),
        "action_id": action_id,
    }
    if full:
        d.update(evidence=f.evidence, details=f.details, explanation=f.explanation,
                 explanation_source=f.explanation_source)
    return d


def ser_run(r: PipelineRun) -> dict:
    return {
        "id": r.id, "asset_id": r.asset_id, "prompt": r.prompt, "spec": r.spec, "planner": r.planner,
        "rule_set_version": r.rule_set_version, "status": r.status, "metrics": r.metrics, "preview": r.preview,
        "output_asset_id": r.output_asset_id, "approved_by": r.approved_by,
        "created_at": r.created_at.isoformat(), "committed_at": r.committed_at.isoformat() if r.committed_at else None,
    }


# ------------------------------------------------------------------ system


def ml_status() -> dict:
    import importlib.metadata as md

    from app.ml.embeddings import get_semantic

    def ver(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    sem = get_semantic()
    return {
        "isolation_forest": {"library": "scikit-learn", "version": ver("scikit-learn"), "ready": ver("scikit-learn") is not None,
                             "use": "anomalous values (fitted per dataset during each scan)"},
        "rapidfuzz": {"library": "rapidfuzz", "version": ver("rapidfuzz"), "ready": ver("rapidfuzz") is not None,
                      "use": "duplicate-record name matching, fuzzy column resolution"},
        "embeddings": {**sem.status(), "library": "sentence-transformers", "version": ver("sentence-transformers"),
                       "ready": sem.installed and ver("sentence-transformers") is not None,
                       "use": "reworded duplicate documents, header matching across datasets, planner column matching"},
    }


@router.get("/health")
def health():
    return {"status": "ok", "llm": get_gateway().status(), "ml": ml_status(), "operators": list(CATALOG)}


@router.post("/demo/setup")
def demo_setup(bg: BackgroundTasks, db: Session = Depends(get_db)):
    created = W.setup_demo(db)
    scans = []
    for ws in created:
        sc = Scan(workspace_id=ws.id)
        db.add(sc)
        db.flush()
        scans.append(sc.id)
    db.commit()
    for sid in scans:
        bg.add_task(run_scan, sid)
    return {"workspaces": [{"id": w.id, "name": w.name} for w in created], "scans": scans}


# ------------------------------------------------------------------ workspaces


class WorkspaceIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    industry: str = ""


@router.get("/workspaces")
def list_workspaces(db: Session = Depends(get_db)):
    out = []
    for w in db.scalars(select(Workspace).order_by(Workspace.created_at)):
        sc = latest_scan(db, w.id)
        out.append({"id": w.id, "name": w.name, "industry": w.industry, "is_demo": w.is_demo,
                    "last_scan": sc.finished_at.isoformat() if sc and sc.finished_at else None})
    return out


@router.post("/workspaces")
def create_ws(body: WorkspaceIn, db: Session = Depends(get_db), actor: str = Depends(actor_of)):
    ws = W.create_workspace(db, body.name, body.industry, actor=actor)
    return {"id": ws.id, "name": ws.name}


@router.delete("/workspaces/{ws_id}")
def delete_ws(ws: Workspace = Depends(get_ws), db: Session = Depends(get_db)):
    W.delete_workspace(db, ws.id)
    return {"deleted": ws.id}


@router.get("/workspaces/{ws_id}/summary")
def summary(ws: Workspace = Depends(get_ws), db: Session = Depends(get_db)):
    data = compute_summary(db, ws.id, get_settings())
    running = db.scalars(select(Scan).where(Scan.workspace_id == ws.id, Scan.status.in_(["queued", "running"]))
                         .order_by(Scan.started_at.desc())).first()
    data["running_scan"] = {"id": running.id, "stage": running.stage, "progress": running.progress} if running else None
    data["workspace"] = {"id": ws.id, "name": ws.name, "industry": ws.industry}
    top = []
    if data["has_scan"]:
        top = db.scalars(select(Finding).where(Finding.workspace_id == ws.id, Finding.scan_id == data["scan"]["id"],
                                               Finding.status == "open").order_by(Finding.priority.desc()).limit(6))
    data["top_findings"] = [ser_finding(f) for f in top]
    return data


# ------------------------------------------------------------------ ingest


@router.post("/workspaces/{ws_id}/upload")
async def upload(files: list[UploadFile] = File(...), ws: Workspace = Depends(get_ws), db: Session = Depends(get_db),
                 actor: str = Depends(actor_of)):
    lay = Layout(ws.id).ensure()
    saved = []
    for uf in files:
        try:
            name = safe_name(uf.filename or "upload.bin")
        except ValueError as e:
            bad_request(e)
        tmp = lay.staging / f"upload_{name}"
        with tmp.open("wb") as fh:
            shutil.copyfileobj(uf.file, fh)
        if name.lower().endswith(".zip"):
            try:
                saved += safe_extract_zip(tmp, lay.files)
            except Exception as e:  # noqa: BLE001
                tmp.unlink(missing_ok=True)
                bad_request(ValueError(f"could not extract {name}: {e}"))
            tmp.unlink(missing_ok=True)
        else:
            dest = lay.files / name
            stem, _, ext = name.rpartition(".")
            n = 1
            while dest.exists():
                dest = lay.files / (f"{stem} ({n}).{ext}" if stem else f"{name} ({n})")
                n += 1
            shutil.move(str(tmp), dest)
            saved.append(dest.name)
    audit(db, ws.id, actor, "files.uploaded", "workspace", ws.id, {"count": len(saved)})
    return {"saved": saved}


@router.post("/workspaces/{ws_id}/connectors/{kind}")
async def upload_connector(kind: Literal["s3_inventory", "query_log", "access_manifest"], file: UploadFile = File(...),
                           ws: Workspace = Depends(get_ws), db: Session = Depends(get_db), actor: str = Depends(actor_of)):
    lay = Layout(ws.id).ensure()
    target = {"s3_inventory": "s3_inventory.csv", "query_log": "query_log.csv", "access_manifest": "access_manifest.json"}[kind]
    with (lay.connectors / target).open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    audit(db, ws.id, actor, "connector.updated", "connector", kind, {"file": file.filename})
    return {"connector": kind, "saved": target}


@router.get("/workspaces/{ws_id}/connectors")
def connectors(ws: Workspace = Depends(get_ws), db: Session = Depends(get_db)):
    lay = Layout(ws.id).ensure()
    counts = dict(db.execute(select(Asset.source, func.count()).where(Asset.workspace_id == ws.id).group_by(Asset.source)).all())
    return [
        {"id": "local", "name": "Uploaded files", "type": "Files (CSV, Excel, JSON, PDF, DOCX, TXT, ZIP, SQLite)",
         "connected": any(lay.files.iterdir()), "objects": counts.get("upload", 0), "mode": "content scan"},
        {"id": "s3_inventory", "name": "AWS S3 (Inventory report)", "type": "CSV: bucket,key,size,last_modified,etag,storage_class",
         "connected": (lay.connectors / "s3_inventory.csv").exists(), "objects": counts.get("s3-inventory", 0), "mode": "metadata only"},
        {"id": "query_log", "name": "Database query log", "type": "CSV: timestamp,system,query",
         "connected": (lay.connectors / "query_log.csv").exists(), "objects": None, "mode": "column usage"},
        {"id": "access_manifest", "name": "Access & ownership", "type": "JSON from IAM / Drive sharing export",
         "connected": (lay.connectors / "access_manifest.json").exists(), "objects": None, "mode": "exposure"},
    ]


# ------------------------------------------------------------------ scans


@router.post("/workspaces/{ws_id}/scans")
def start_scan(bg: BackgroundTasks, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db),
               actor: str = Depends(actor_of)):
    running = db.scalars(select(Scan).where(Scan.workspace_id == ws.id, Scan.status.in_(["queued", "running"]))).first()
    if running:
        return {"id": running.id, "status": running.status}
    sc = Scan(workspace_id=ws.id)
    db.add(sc)
    db.flush()
    audit(db, ws.id, actor, "scan.started", "scan", sc.id)
    db.commit()
    bg.add_task(run_scan, sc.id)
    return {"id": sc.id, "status": "queued"}


@router.get("/workspaces/{ws_id}/scans/{scan_id}")
def get_scan(scan_id: str, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db)):
    sc = owned(db, Scan, scan_id, ws)
    return {"id": sc.id, "status": sc.status, "stage": sc.stage, "progress": sc.progress, "stats": sc.stats,
            "error": sc.error.splitlines()[0] if sc.error else ""}


# ------------------------------------------------------------------ assets


@router.get("/workspaces/{ws_id}/assets")
def list_assets(ws: Workspace = Depends(get_ws), db: Session = Depends(get_db), status: str | None = None,
                source: str | None = None, q: str | None = None, limit: int = 500):
    stmt = select(Asset).where(Asset.workspace_id == ws.id)
    if status:
        stmt = stmt.where(Asset.status == status)
    if source:
        stmt = stmt.where(Asset.source == source)
    if q:
        stmt = stmt.where(Asset.path.ilike(f"%{q}%"))
    rows = db.scalars(stmt.order_by(Asset.size_bytes.desc()).limit(min(limit, 5000))).all()
    flagged: dict[str, list[str]] = {}
    sc = latest_scan(db, ws.id)
    if sc:
        for f in db.scalars(select(Finding).where(Finding.workspace_id == ws.id, Finding.scan_id == sc.id, Finding.status == "open")):
            for t in f.target_asset_ids:
                flagged.setdefault(t, []).append(f.action)
    return [{**ser_asset(a), "flags": sorted(set(flagged.get(a.id, [])))} for a in rows]


class TrashIn(BaseModel):
    asset_ids: list[str] = Field(min_length=1, max_length=500)


@router.post("/workspaces/{ws_id}/assets/trash")
def trash(body: TrashIn, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db), actor: str = Depends(actor_of)):
    """Move files to the restorable trash by hand. Nothing is hard-deleted."""
    assets = [owned(db, Asset, i, ws) for i in dict.fromkeys(body.asset_ids)]
    try:
        acts = trash_assets(db, ws.id, assets, actor)
    except (ActionError, OSError) as e:
        bad_request(e)
    return {"action_ids": [a.id for a in acts], "trashed": len(acts),
            "bytes": sum(a.size_bytes for a in assets)}


@router.post("/workspaces/{ws_id}/assets/{asset_id}/restore")
def restore(asset_id: str, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db), actor: str = Depends(actor_of)):
    a = owned(db, Asset, asset_id, ws)
    try:
        act = restore_asset(db, ws.id, a, actor)
    except (ActionError, OSError) as e:
        bad_request(e)
    return {"restored": a.id, "action_id": act.id}


@router.get("/workspaces/{ws_id}/assets/{asset_id}/profile")
def asset_profile(asset_id: str, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db)):
    a = owned(db, Asset, asset_id, ws)
    try:
        v = P.load_view(db, a)
    except ActionError as e:
        bad_request(e)
    from app.pipeline.planner import suggest_prompt

    prompt = suggest_prompt(v)
    scan = latest_scan(db, ws.id)
    if scan:  # personal data inside free text is found by the scan, not the profiler: fold its fix in
        for f in db.scalars(select(Finding).where(Finding.workspace_id == ws.id, Finding.scan_id == scan.id,
                                                  Finding.status == "open", Finding.kind == "pii_in_text")):
            extra = (f.details or {}).get("suggested_prompt")
            if a.id in f.target_asset_ids and extra and extra.lower() not in prompt.lower():
                prompt = f"{prompt.rstrip('.')}. {extra}"
    return {"asset": ser_asset(a), "rows": v.rows, "columns": [p.to_dict() for p in v.profiles.values()],
            "issues": [i.__dict__ for i in v.issues], "suggested_prompt": prompt}


# ------------------------------------------------------------------ findings & actions


@router.get("/workspaces/{ws_id}/findings")
def list_findings(ws: Workspace = Depends(get_ws), db: Session = Depends(get_db), status: str | None = "open",
                  action: str | None = None, severity: str | None = None, category: str | None = None):
    sc = latest_scan(db, ws.id)
    if not sc:
        return []
    stmt = select(Finding).where(Finding.workspace_id == ws.id, Finding.scan_id == sc.id)
    if status and status != "all":
        stmt = stmt.where(Finding.status == status)
    if action:
        stmt = stmt.where(Finding.action == action)
    if severity:
        stmt = stmt.where(Finding.severity == severity)
    if category:
        stmt = stmt.where(Finding.category == category)
    acts = applied_actions(db, ws.id)
    out = []
    for f in db.scalars(stmt.order_by(Finding.priority.desc())):
        d = ser_finding(f, action_id=acts.get(f.id))
        first = db.get(Asset, f.target_asset_ids[0]) if f.target_asset_ids else None
        d["source"] = first.source if first else ""
        out.append(d)
    return out


@router.get("/workspaces/{ws_id}/findings/{finding_id}")
def get_finding(finding_id: str, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db)):
    f = owned(db, Finding, finding_id, ws)
    ids = set(f.target_asset_ids) | set(f.keep_asset_ids) | {p["asset_id"] for p in f.details.get("plan", [])}
    assets = {i: ser_asset(a) for i in ids if (a := db.get(Asset, i)) is not None}
    actions = db.scalars(select(Action).where(Action.finding_id == f.id).order_by(Action.created_at.desc())).all()
    return {**ser_finding(f, full=True), "assets": assets,
            "actions": [{"id": x.id, "action": x.action, "status": x.status, "actor": x.actor, "note": x.note,
                         "created_at": x.created_at.isoformat()} for x in actions]}


@router.post("/workspaces/{ws_id}/findings/{finding_id}/explain")
def explain_finding(finding_id: str, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db), refresh: bool = False):
    f = owned(db, Finding, finding_id, ws)
    if not f.explanation or refresh:
        f.explanation, f.explanation_source = explain(f)
    return {"explanation": f.explanation, "source": f.explanation_source}


class ActionIn(BaseModel):
    action: Literal["APPLY", "ARCHIVE", "DELETE", "SECURE", "KEEP", "IGNORE"]
    confirm: bool = False


@router.post("/workspaces/{ws_id}/findings/{finding_id}/actions")
def act(finding_id: str, body: ActionIn, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db),
        actor: str = Depends(actor_of)):
    f = owned(db, Finding, finding_id, ws)
    if body.action in ("APPLY", "ARCHIVE", "DELETE", "SECURE") and not body.confirm:
        raise HTTPException(409, "confirmation required for this action")
    try:
        a = apply_action(db, f, body.action, actor)
    except (ActionError, ValueError) as e:
        bad_request(e)
    return {"action_id": a.id, "action": a.action, "note": a.note, "finding_status": f.status}


class BulkIn(BaseModel):
    min_confidence: float = Field(0.95, ge=0.5, le=1.0)
    confirm: bool = False


BULK_ACTIONS = ("ARCHIVE", "DELETE", "SECURE")


@router.post("/workspaces/{ws_id}/findings/bulk-apply")
def bulk_apply(body: BulkIn, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db), actor: str = Depends(actor_of)):
    """Approve every open, high-confidence ARCHIVE/DELETE/SECURE recommendation (each stays undoable)."""
    sc = latest_scan(db, ws.id)
    if not sc:
        return {"applied": 0, "action_ids": []}
    candidates = list(db.scalars(select(Finding).where(
        Finding.workspace_id == ws.id, Finding.scan_id == sc.id, Finding.status == "open",
        Finding.action.in_(BULK_ACTIONS), Finding.confidence >= body.min_confidence).order_by(Finding.priority.desc())))
    if not body.confirm:
        return {"would_apply": len(candidates), "savings_inr_year": round(sum(f.savings_inr_year for f in candidates)),
                "titles": [f.title for f in candidates[:20]]}
    ids, skipped = [], 0
    for f in candidates:
        db.refresh(f)
        if f.status != "open":
            skipped += 1
            continue
        try:
            ids.append(apply_action(db, f, "APPLY", actor).id)
        except (ActionError, ValueError):
            skipped += 1
    audit(db, ws.id, actor, "findings.bulk_apply", "workspace", ws.id,
          {"applied": len(ids), "skipped": skipped, "min_confidence": body.min_confidence})
    return {"applied": len(ids), "skipped": skipped, "action_ids": ids}


@router.get("/workspaces/{ws_id}/security")
def security(ws: Workspace = Depends(get_ws), db: Session = Depends(get_db)):
    from app.scanner.pii import ENTITIES

    sc = latest_scan(db, ws.id)
    if not sc:
        return {"has_scan": False}
    acts = applied_actions(db, ws.id)
    fs = list(db.scalars(select(Finding).where(Finding.workspace_id == ws.id, Finding.scan_id == sc.id,
                                               Finding.kind.in_(["sensitive_data", "secret_exposed"]))
                         .order_by(Finding.risk_score.desc())))
    totals: dict[str, int] = {}
    rows = []
    for f in fs:
        a = db.get(Asset, f.target_asset_ids[0])
        if f.kind == "sensitive_data":
            ents = f.details.get("entities", {})
            for k, v in ents.items():
                totals[ENTITIES[k].label] = totals.get(ENTITIES[k].label, 0) + v
            pii_labels = [ENTITIES[k].label for k in sorted(ents, key=lambda k: -ENTITIES[k].sensitivity)]
        else:
            ents = f.details.get("secrets", {})
            totals["Credentials & keys"] = totals.get("Credentials & keys", 0) + sum(ents.values())
            pii_labels = list(ents)
        rows.append({
            "finding_id": f.id, "kind": f.kind, "file": a.name if a else "?", "path": a.path if a else "",
            "source": a.source if a else "", "pii": pii_labels, "records": f.records_affected,
            "shared_with": a.shared_with if a else None, "encrypted": a.encrypted if a else None,
            "risk": f.risk_score, "severity": f.severity, "action": f.action, "status": f.status,
            "resolution": f.resolution, "action_id": acts.get(f.id),
            "past_retention": f.details.get("past_retention", False),
        })
    s = get_settings()
    open_rows = [r for r in rows if r["status"] == "open"]
    return {
        "has_scan": True,
        "entities": [{"label": k, "count": v} for k, v in sorted(totals.items(), key=lambda kv: -kv[1])],
        "exposed_count": sum(1 for r in open_rows if (r["shared_with"] or 0) > s.exposure_users),
        "open_count": len(open_rows),
        "secured_count": sum(1 for r in rows if r["resolution"] in ("SECURE", "DELETE")),
        "risky_records": sum(r["records"] for r in open_rows if r["kind"] == "sensitive_data"),
        "exposure_threshold": s.exposure_users,
        "rows": rows,
    }


class DecisionsIn(BaseModel):
    decisions: dict[str, Literal["merge", "keep_both", "ignore"]]


@router.post("/workspaces/{ws_id}/findings/{finding_id}/row-decisions")
def row_decisions(finding_id: str, body: DecisionsIn, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db),
                  actor: str = Depends(actor_of)):
    f = owned(db, Finding, finding_id, ws)
    if f.kind != "duplicate_rows":
        bad_request(ValueError("not a duplicate-rows finding"))
    details = dict(f.details)
    clusters = [dict(c) for c in details.get("clusters", [])]
    for c in clusters:
        if c["cluster_id"] in body.decisions:
            dec = body.decisions[c["cluster_id"]]
            if dec == "merge" and c.get("conflict_columns"):
                bad_request(ValueError(f"group {c['cluster_id']} has conflicting {', '.join(c['conflict_columns'])}; it cannot be merged"))
            c["decision"] = dec
    details["clusters"] = clusters
    f.details = details
    audit(db, ws.id, actor, "rows.reviewed", "finding", f.id, {"decisions": len(body.decisions)})
    counts = {k: sum(1 for c in clusters if c.get("decision") == k) for k in ("merge", "keep_both", "ignore")}
    return {"saved": len(body.decisions), "counts": counts}


@router.get("/workspaces/{ws_id}/actions")
def list_actions(ws: Workspace = Depends(get_ws), db: Session = Depends(get_db)):
    rows = db.scalars(select(Action).where(Action.workspace_id == ws.id).order_by(Action.created_at.desc()).limit(200))
    out = []
    for a in rows:
        f = db.get(Finding, a.finding_id) if a.finding_id else None
        out.append({"id": a.id, "action": a.action, "status": a.status, "actor": a.actor, "note": a.note,
                    "finding_id": a.finding_id, "finding_title": f.title if f else "",
                    "assets": [x.get("name") for x in a.affected if x.get("name")],
                    "created_at": a.created_at.isoformat(), "undone_at": a.undone_at.isoformat() if a.undone_at else None})
    return out


@router.post("/workspaces/{ws_id}/actions/{action_id}/undo")
def undo(action_id: str, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db), actor: str = Depends(actor_of)):
    a = owned(db, Action, action_id, ws)
    try:
        undo_action(db, a, actor)
    except (ActionError, OSError) as e:
        bad_request(e)
    return {"undone": a.id}


@router.get("/workspaces/{ws_id}/audit")
def audit_log(ws: Workspace = Depends(get_ws), db: Session = Depends(get_db), limit: int = 200):
    rows = db.scalars(select(AuditEvent).where(AuditEvent.workspace_id == ws.id)
                      .order_by(AuditEvent.created_at.desc()).limit(min(limit, 1000)))
    return [{"id": e.id, "actor": e.actor, "event": e.event, "resource_type": e.resource_type,
             "resource_id": e.resource_id, "data": e.data, "created_at": e.created_at.isoformat()} for e in rows]


# ------------------------------------------------------------------ rules


@router.get("/workspaces/{ws_id}/rules")
def get_rules(ws: Workspace = Depends(get_ws), db: Session = Depends(get_db)):
    versions = db.scalars(select(RuleSet).where(RuleSet.workspace_id == ws.id).order_by(RuleSet.version.desc())).all()
    active = next((v for v in versions if v.active), None)
    return {"active": {"version": active.version, "rules": active.rules, "created_by": active.created_by,
                       "created_at": active.created_at.isoformat()} if active else None,
            "history": [{"version": v.version, "created_by": v.created_by, "created_at": v.created_at.isoformat(),
                         "active": v.active} for v in versions]}


class RulesIn(BaseModel):
    rules: dict


@router.put("/workspaces/{ws_id}/rules")
def put_rules(body: RulesIn, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db), actor: str = Depends(actor_of)):
    r = body.rules
    if not isinstance(r.get("dedupe", {}), dict) or not isinstance(r.get("required_fields", []), list):
        bad_request(ValueError("rules must include dedupe (object) and required_fields (list)"))
    rs = W.update_rules(db, ws.id, r, actor)
    return {"version": rs.version}


# ------------------------------------------------------------------ pipelines (Clean Studio)


class PlanIn(BaseModel):
    asset_id: str
    prompt: str = Field(min_length=3, max_length=2000)
    use_review: bool = True


@router.post("/workspaces/{ws_id}/pipelines/plan")
def plan(body: PlanIn, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db)):
    a = owned(db, Asset, body.asset_id, ws)
    try:
        return P.plan_pipeline(db, ws.id, a, body.prompt, body.use_review)
    except ActionError as e:
        bad_request(e)


class DryRunIn(BaseModel):
    asset_id: str
    spec: dict
    prompt: str = ""
    planner: Literal["llm", "rules", "manual"] = "manual"


@router.post("/workspaces/{ws_id}/pipelines/validate")
def validate(body: DryRunIn, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db)):
    from app.pipeline.validator import graph, validate_spec

    a = owned(db, Asset, body.asset_id, ws)
    try:
        v = P.load_view(db, a)
    except ActionError as e:
        bad_request(e)
    rep = validate_spec({**body.spec, "source_asset_id": a.id}, list(v.df.columns), v.profiles)
    return {"validation": rep.to_dict(), "spec": rep.spec.model_dump() if rep.spec else body.spec,
            "graph": graph(rep.spec, a.name) if rep.spec else None}


@router.post("/workspaces/{ws_id}/pipelines/dry-run")
def dry_run(body: DryRunIn, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db), actor: str = Depends(actor_of)):
    a = owned(db, Asset, body.asset_id, ws)
    try:
        run = P.dry_run(db, ws.id, a, body.spec, body.prompt, body.planner, actor)
    except ActionError as e:
        bad_request(e)
    return ser_run(run)


class CommitIn(BaseModel):
    mode: Literal["new_file", "in_place"] = "new_file"


@router.post("/workspaces/{ws_id}/runs/{run_id}/commit")
def commit(run_id: str, body: CommitIn | None = Body(default=None), ws: Workspace = Depends(get_ws),
           db: Session = Depends(get_db), actor: str = Depends(actor_of)):
    r = owned(db, PipelineRun, run_id, ws)
    vc = None
    try:
        if body and body.mode == "in_place":
            vc = VAULT.commit_in_place(db, r, actor)
        else:
            P.commit_run(db, r, actor)
    except ActionError as e:
        bad_request(e)
    return {**ser_run(r), "vault_commit": VAULT.ser_commit(vc) if vc else None}


@router.post("/workspaces/{ws_id}/runs/{run_id}/discard")
def discard(run_id: str, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db), actor: str = Depends(actor_of)):
    r = owned(db, PipelineRun, run_id, ws)
    try:
        P.discard_run(db, r, actor)
    except ActionError as e:
        bad_request(e)
    return ser_run(r)


@router.get("/workspaces/{ws_id}/runs/{run_id}/diff")
def run_diff(run_id: str, op: Literal["DELETE", "QUARANTINE", "REDACT", "NORMALIZE"] | None = None, offset: int = 0,
             limit: int = 50, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db)):
    """Diff Inspector: affected rows of the virtual data tree, git-diff style."""
    from app.pipeline import virtual as V

    r = owned(db, PipelineRun, run_id, ws)
    try:
        tree = P.mount_run(db, r)
    except ActionError as e:
        bad_request(e)
    return V.inspector(tree, op, max(0, offset), max(1, min(limit, 200)))


@router.get("/workspaces/{ws_id}/runs/{run_id}/impact")
def run_impact(run_id: str, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db)):
    """Simulate downstream impact: what analytics reading this file would notice after the commit."""
    from app.pipeline import virtual as V

    r = owned(db, PipelineRun, run_id, ws)
    try:
        tree = P.mount_run(db, r)
    except ActionError as e:
        bad_request(e)
    return V.impact(tree, r.spec["steps"])


@router.get("/workspaces/{ws_id}/runs/{run_id}/patch")
def run_patch(run_id: str, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db)):
    """The Data-Patch as a JSON file (identifiers masked)."""
    r = owned(db, PipelineRun, run_id, ws)
    path = Layout(ws.id).staging / r.id / "data-patch.json"
    if not path.exists():
        raise HTTPException(404, "no Data-Patch for this run (it was discarded, or predates the virtual data tree)")
    return FileResponse(path, media_type="application/json", filename=f"data-patch-{r.id}.json")


@router.get("/workspaces/{ws_id}/commits")
def list_commits(ws: Workspace = Depends(get_ws), db: Session = Depends(get_db)):
    rows = db.scalars(select(VaultCommit).where(VaultCommit.workspace_id == ws.id).order_by(VaultCommit.created_at.desc()).limit(100))
    return [VAULT.ser_commit(c) for c in rows]


@router.post("/workspaces/{ws_id}/commits/{commit_id}/rollback")
def rollback_commit(commit_id: str, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db), actor: str = Depends(actor_of)):
    vc = owned(db, VaultCommit, commit_id, ws)
    try:
        VAULT.rollback(db, vc, actor)
    except ActionError as e:
        bad_request(e)
    return VAULT.ser_commit(vc)


@router.get("/workspaces/{ws_id}/commits/{commit_id}/verify")
def verify_commit(commit_id: str, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db)):
    return VAULT.verify(owned(db, VaultCommit, commit_id, ws))


@router.get("/workspaces/{ws_id}/runs")
def runs(ws: Workspace = Depends(get_ws), db: Session = Depends(get_db)):
    rows = db.scalars(select(PipelineRun).where(PipelineRun.workspace_id == ws.id).order_by(PipelineRun.created_at.desc()).limit(50))
    out = []
    for r in rows:
        d = ser_run(r)
        d.pop("preview")
        a = db.get(Asset, r.asset_id)
        d["asset_name"] = a.name if a else ""
        out.append(d)
    return out


@router.get("/workspaces/{ws_id}/runs/{run_id}")
def get_run(run_id: str, ws: Workspace = Depends(get_ws), db: Session = Depends(get_db)):
    return ser_run(owned(db, PipelineRun, run_id, ws))


# ------------------------------------------------------------------ reports & observability


@router.get("/workspaces/{ws_id}/reports/dpdp")
def dpdp(ws: Workspace = Depends(get_ws), db: Session = Depends(get_db)):
    return {**W.dpdp_report(db, ws.id), "workspace": {"id": ws.id, "name": ws.name, "industry": ws.industry}}


@router.get("/workspaces/{ws_id}/ai-usage")
def ai_usage(ws: Workspace = Depends(get_ws), db: Session = Depends(get_db)):
    rows = db.scalars(select(LLMCall).where(LLMCall.workspace_id == ws.id).order_by(LLMCall.created_at.desc()).limit(100)).all()
    return {"status": get_gateway().status(), "calls": [
        {"purpose": c.purpose, "model": c.model, "ok": c.ok, "latency_ms": c.latency_ms, "input_tokens": c.input_tokens,
         "output_tokens": c.output_tokens, "error": c.error, "created_at": c.created_at.isoformat()} for c in rows]}
