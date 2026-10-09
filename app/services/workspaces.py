"""Workspace lifecycle, tenant rules, demo setup and the DPDP readiness report."""
from __future__ import annotations

import shutil

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.engine.health import latest_scan
from app.models import (Action, Asset, AuditEvent, Finding, HealthSnapshot, LLMCall, PipelineRun, RuleSet, Scan,
                        VaultCommit, Workspace)
from app.services.actions import audit
from app.services.storage import Layout

DEFAULT_RULES = {
    "country": "IN",
    "date_format": "%Y-%m-%d",
    "required_fields": ["email"],
    "formats": {"email": "email", "phone": "phone"},
    "allowed_values": {},
    "unique": [],
    "dedupe": {"keys": ["email"], "keep": "most_complete", "never_merge_on_conflict": []},
    "retention_days": 1095,
}

DEMO_TENANTS = [
    {
        "slug": "acme", "name": "Acme Retail Pvt Ltd", "industry": "Retail & e-commerce",
        "rules": {**DEFAULT_RULES,
                  "dedupe": {"keys": ["email"], "keep": "most_complete", "never_merge_on_conflict": ["pan"]}},
    },
    {
        "slug": "zenith", "name": "Zenith Finance Ltd", "industry": "Lending (NBFC)",
        "rules": {**DEFAULT_RULES,
                  "required_fields": ["pan", "email"],
                  "formats": {"email": "email", "phone": "phone", "pan": "pan"},
                  "unique": ["pan"],
                  "dedupe": {"keys": ["pan"], "keep": "most_complete", "never_merge_on_conflict": ["aadhaar"]}},
    },
]


def create_workspace(db: Session, name: str, industry: str = "", rules: dict | None = None,
                     is_demo: bool = False, actor: str = "admin") -> Workspace:
    ws = Workspace(name=name, industry=industry, is_demo=is_demo)
    db.add(ws)
    db.flush()
    db.add(RuleSet(workspace_id=ws.id, version=1, rules=rules or DEFAULT_RULES, created_by=actor))
    Layout(ws.id).ensure()
    audit(db, ws.id, actor, "workspace.created", "workspace", ws.id, {"name": name})
    return ws


def update_rules(db: Session, ws_id: str, rules: dict, actor: str) -> RuleSet:
    current = db.scalars(select(RuleSet).where(RuleSet.workspace_id == ws_id).order_by(RuleSet.version.desc())).first()
    version = (current.version if current else 0) + 1
    for rs in db.scalars(select(RuleSet).where(RuleSet.workspace_id == ws_id, RuleSet.active.is_(True))):
        rs.active = False
    rs = RuleSet(workspace_id=ws_id, version=version, rules=rules, created_by=actor, active=True)
    db.add(rs)
    audit(db, ws_id, actor, "rules.updated", "rule_set", str(version), {"version": version})
    return rs


def delete_workspace(db: Session, ws_id: str) -> None:
    for model in (Action, AuditEvent, VaultCommit, PipelineRun, HealthSnapshot, Finding, Scan, Asset, RuleSet, LLMCall):
        db.execute(delete(model).where(model.workspace_id == ws_id))
    db.execute(delete(Workspace).where(Workspace.id == ws_id))
    shutil.rmtree(Layout(ws_id).root, ignore_errors=True)


def setup_demo(db: Session) -> list[Workspace]:
    """(Re)create the demo tenants from backend/demo/<slug>. Non-demo workspaces are untouched."""
    from scripts.generate_demo import ensure_demo  # local import: dev tooling

    ensure_demo(get_settings().demo_dir)
    for ws in db.scalars(select(Workspace).where(Workspace.is_demo.is_(True))):
        delete_workspace(db, ws.id)
    db.flush()
    created = []
    for t in DEMO_TENANTS:
        ws = create_workspace(db, t["name"], t["industry"], t["rules"], is_demo=True, actor="demo")
        src = get_settings().demo_dir / t["slug"]
        lay = Layout(ws.id).ensure()
        shutil.rmtree(lay.files)
        shutil.copytree(src / "files", lay.files)
        if (src / "connectors").exists():
            shutil.copytree(src / "connectors", lay.connectors, dirs_exist_ok=True)
        created.append(ws)
    return created


def dpdp_report(db: Session, ws_id: str) -> dict:
    s = get_settings()
    scan = latest_scan(db, ws_id)
    if not scan:
        return {"has_scan": False}
    fs = list(db.scalars(select(Finding).where(Finding.workspace_id == ws_id, Finding.scan_id == scan.id,
                                               Finding.kind.in_(["sensitive_data", "secret_exposed"]))))
    inventory = []
    for f in fs:
        a = db.get(Asset, f.target_asset_ids[0])
        inventory.append({
            "finding_id": f.id, "asset": a.name if a else "?", "path": a.path if a else "", "kind": f.kind,
            "entities": f.details.get("entities") or f.details.get("secrets"), "records": f.records_affected,
            "shared_with": a.shared_with if a else None, "encrypted": a.encrypted if a else None,
            "days_since_use": f.details.get("days_since_use"), "past_retention": f.details.get("past_retention", False),
            "risk": f.risk_score, "severity": f.severity, "status": f.status, "resolution": f.resolution,
            "recommended": f.action,
        })
    open_pii = [i for i in inventory if i["status"] == "open" and i["kind"] == "sensitive_data"]
    checks = [
        {"id": "inventory", "title": "Personal data inventory is maintained",
         "detail": f"{len([i for i in inventory if i['kind'] == 'sensitive_data'])} datasets with personal data identified by automated scan",
         "passed": True, "ref": "DPDP Act s.8 (Data Fiduciary obligations)"},
        {"id": "exposure", "title": "Personal data is not over-exposed",
         "detail": f"{sum(1 for i in open_pii if (i['shared_with'] or 0) > s.exposure_users)} open datasets shared with more than {s.exposure_users} users",
         "passed": not any((i["shared_with"] or 0) > s.exposure_users for i in open_pii), "ref": "s.8(5) reasonable security safeguards"},
        {"id": "encryption", "title": "Personal data is encrypted at rest",
         "detail": f"{sum(1 for i in open_pii if not i['encrypted'])} open datasets stored unencrypted",
         "passed": not any(not i["encrypted"] for i in open_pii), "ref": "Rule 6 (security safeguards)"},
        {"id": "retention", "title": "Data is erased once its purpose is served",
         "detail": f"{sum(1 for i in open_pii if i['past_retention'])} datasets unused for more than {s.retention_days} days",
         "passed": not any(i["past_retention"] for i in open_pii), "ref": "s.8(7), Rule 8 & Third Schedule (erasure)"},
        {"id": "secrets", "title": "No credentials in shared files",
         "detail": f"{sum(1 for i in inventory if i['kind'] == 'secret_exposed' and i['status'] == 'open')} open credential exposures",
         "passed": not any(i["kind"] == "secret_exposed" and i["status"] == "open" for i in inventory), "ref": "Rule 6 (access control)"},
        {"id": "audit", "title": "Processing actions are logged",
         "detail": "Every scan, action, approval and pipeline run is recorded in the audit trail",
         "passed": True, "ref": "Rule 6 (logs and monitoring)"},
    ]
    passed = sum(1 for c in checks if c["passed"])
    return {
        "has_scan": True, "scan_finished_at": scan.finished_at.isoformat() if scan.finished_at else None,
        "readiness_pct": round(100 * passed / len(checks)), "checks": checks, "inventory": inventory,
        "total_records": sum(i["records"] for i in inventory if i["kind"] == "sensitive_data"),
        "disclaimer": "Automated technical readiness indicators, not legal advice. The DPDP Rules 2025 were notified on "
                      "13 Nov 2025; most obligations apply from ~May 2027. Verify against the Gazette text.",
    }
