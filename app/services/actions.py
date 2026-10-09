"""Apply and undo recommendations. Nothing is ever hard-deleted: DELETE moves to a restorable
trash, ARCHIVE moves to cold storage, SECURE moves to an owner-only vault. Every change is
recorded with enough state to undo it."""
from __future__ import annotations

import csv
import os
import shutil
import sqlite3
import stat

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.engine.health import latest_scan, record_snapshot
from app.models import Action, Asset, AuditEvent, Finding, utcnow
from app.services.storage import Layout, asset_path

USER_ACTIONS = {"APPLY", "ARCHIVE", "DELETE", "SECURE", "KEEP", "IGNORE"}


class ActionError(ValueError):
    pass


def audit(db: Session, ws: str, actor: str, event: str, rtype: str, rid: str, data: dict | None = None) -> None:
    db.add(AuditEvent(workspace_id=ws, actor=actor, event=event, resource_type=rtype, resource_id=rid, data=data or {}))


def _move(layout: Layout, a: Asset, dest_base, new_status: str, storage_class: str | None = None) -> dict:
    rec = {"asset_id": a.id, "name": a.name, "prev_status": a.status, "prev_storage_class": a.storage_class,
           "prev_shared_with": a.shared_with, "prev_encrypted": a.encrypted, "new_status": new_status}
    rec["prev_location"] = (a.meta or {}).get("location")
    if a.source in ("upload", "pipeline-output"):
        src = asset_path(layout, a)
        dst = layout.inside(dest_base, a.path)
        if not src.exists():
            raise ActionError(f"{a.path} is no longer on disk")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        rec.update(moved_from=str(src), moved_to=str(dst))
        meta = {k: v for k, v in (a.meta or {}).items() if k != "location"}
        if new_status == "secured":
            os.chmod(dst, stat.S_IRUSR | stat.S_IWUSR)
            meta["location"] = "secured"
        a.meta = meta
    else:
        rec["simulated"] = True  # metadata-only connector: lifecycle change recorded, write-back is the connector's job
    a.status = new_status
    if storage_class:
        a.storage_class = storage_class
    return rec


def _archive_columns(layout: Layout, a: Asset, f: Finding) -> dict:
    table = f.details["table"]
    cols = [c["column"] for c in f.details["columns"]]
    src = layout.inside(layout.files, a.path)
    out = layout.inside(layout.archive, f"{a.path}__{table}__columns.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    pk = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")') if r[5]] or ["rowid"]
    sel = ", ".join(f'"{c}"' for c in pk + cols)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(pk + cols)
        for row in con.execute(f'SELECT {sel} FROM "{table}"'):
            w.writerow(row)
    con.close()
    return {"asset_id": a.id, "name": a.name, "column_export": str(out), "table": table, "columns": cols,
            "prev_status": a.status, "new_status": a.status}


def _auto_resolve(db: Session, ws: str, exclude_id: str) -> list[str]:
    """Close other open findings whose every target is no longer active (e.g. a PII finding on a deleted copy)."""
    scan = latest_scan(db, ws)
    if not scan:
        return []
    closed = []
    for f in db.scalars(select(Finding).where(Finding.workspace_id == ws, Finding.scan_id == scan.id, Finding.status == "open")):
        if f.id == exclude_id or not f.target_asset_ids or f.kind == "unused_columns":
            continue
        targets = [db.get(Asset, t) for t in f.target_asset_ids]
        if all(t is None or t.status in ("trashed", "archived") for t in targets):
            f.status, f.resolution = "resolved", "AUTO"
            closed.append(f.id)
    return closed


def apply_action(db: Session, f: Finding, action: str, actor: str) -> Action:
    if action not in USER_ACTIONS:
        raise ActionError(f"unsupported action {action}")
    if f.status != "open":
        raise ActionError(f"finding is already {f.status}")
    ws = f.workspace_id
    layout = Layout(ws).ensure()

    if f.action == "KEEP" and action == "APPLY":  # informational findings: applying = acknowledging
        action = "KEEP"
    if action in ("KEEP", "IGNORE"):
        f.status, f.resolution = "dismissed", action
        act = Action(workspace_id=ws, finding_id=f.id, action=action, actor=actor, affected=[],
                     note="Recommendation dismissed; it will stay dismissed on future scans.")
        db.add(act)
        db.flush()
        audit(db, ws, actor, f"finding.{action.lower()}", "finding", f.id, {"title": f.title})
        record_snapshot(db, ws, get_settings(), f"{action.title()}: {f.title}")
        return act

    if f.action == "CLEAN" and action == "APPLY":
        raise ActionError("CLEAN recommendations run through Clean Studio (plan → dry run → approve)")

    affected: list[dict] = []
    if f.kind == "unused_columns":
        if action not in ("APPLY", "ARCHIVE"):
            raise ActionError("unused columns can only be archived")
        affected.append(_archive_columns(layout, db.get(Asset, f.target_asset_ids[0]), f))
        note = "Column values exported to the archive. Dropping the columns from the live database needs DBA approval."
        resolution = "ARCHIVE"
    else:
        if action == "APPLY":
            plan = [p for p in f.details.get("plan", []) if p["action"] in ("DELETE", "ARCHIVE", "SECURE")]
            if not plan:
                plan = [{"asset_id": t, "action": f.action} for t in f.target_asset_ids]
            resolution = f.action
        else:
            plan = [{"asset_id": t, "action": action} for t in f.target_asset_ids]
            resolution = action
        for p in plan:
            a = db.get(Asset, p["asset_id"])
            if a is None or a.workspace_id != ws:
                raise ActionError("asset not found in this workspace")
            if a.status != "active":
                continue
            if p["action"] == "DELETE":
                affected.append(_move(layout, a, layout.trash, "trashed"))
            elif p["action"] == "ARCHIVE":
                affected.append(_move(layout, a, layout.archive, "archived", "DEEP_ARCHIVE"))
            elif p["action"] == "SECURE":
                rec = _move(layout, a, layout.secured, "secured")
                rec["new_shared_with"] = 1
                a.shared_with, a.encrypted = 1, True
                a.status = "active"  # still live data, now owner-only
                rec["new_status"] = "active"
                rec["secured"] = True
                affected.append(rec)
        note = {
            "DELETE": "Moved to trash (restorable). Permanent purge happens only after the retention window.",
            "ARCHIVE": "Moved to cold storage (Deep Archive tier). Restorable.",
            "SECURE": "Access restricted to the owner and moved to the encrypted vault.",
        }.get(resolution, "Applied.")
        if not affected:
            raise ActionError("nothing to apply: targets are no longer active")

    f.status, f.resolution = "resolved", resolution
    db.flush()
    auto = _auto_resolve(db, ws, f.id)
    act = Action(workspace_id=ws, finding_id=f.id, action=resolution, actor=actor,
                 affected=affected + ([{"auto_resolved": auto}] if auto else []), note=note)
    db.add(act)
    db.flush()
    audit(db, ws, actor, f"finding.{resolution.lower()}", "finding", f.id,
          {"title": f.title, "assets": [x.get("name") for x in affected if "name" in x], "action_id": act.id,
           "savings_inr_year": f.savings_inr_year})
    record_snapshot(db, ws, get_settings(), f"{resolution.title()}: {f.title}")
    return act


def undo_action(db: Session, act: Action, actor: str) -> Action:
    if act.status != "applied":
        raise ActionError("action was already undone")
    layout = Layout(act.workspace_id)
    for rec in reversed(act.affected):
        if "auto_resolved" in rec:
            for fid in rec["auto_resolved"]:
                f = db.get(Finding, fid)
                if f and f.resolution == "AUTO":
                    f.status, f.resolution = "open", ""
            continue
        if "column_export" in rec:
            try:
                os.remove(rec["column_export"])
            except FileNotFoundError:
                pass
            continue
        a = db.get(Asset, rec["asset_id"])
        if a is None:
            continue
        if rec.get("moved_to"):
            src, dst = rec["moved_to"], rec["moved_from"]
            if os.path.exists(dst):
                raise ActionError(f"cannot restore {a.path}: a file already exists at the original location")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if rec.get("secured"):
                os.chmod(src, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
            shutil.move(src, dst)
        meta = {k: v for k, v in (a.meta or {}).items() if k != "location"}
        if rec.get("prev_location"):
            meta["location"] = rec["prev_location"]
        a.meta = meta
        a.status = rec["prev_status"]
        a.storage_class = rec["prev_storage_class"]
        a.shared_with = rec["prev_shared_with"]
        a.encrypted = rec["prev_encrypted"]
    if act.finding_id:
        f = db.get(Finding, act.finding_id)
        if f:
            f.status, f.resolution = "open", ""
    act.status, act.undone_at = "undone", utcnow()
    audit(db, act.workspace_id, actor, "action.undo", "action", act.id, {"action": act.action, "finding_id": act.finding_id})
    record_snapshot(db, act.workspace_id, get_settings(), f"Undo {act.action.lower()}")
    return act


def trash_assets(db: Session, ws: str, assets: list[Asset], actor: str) -> list[Action]:
    """Delete files by hand (no recommendation needed). Like every DELETE here, the file only moves to the
    restorable trash; each file gets its own action so it can be restored on its own."""
    layout = Layout(ws).ensure()
    acts = []
    for a in assets:
        if a.status != "active":
            raise ActionError(f"{a.path} is {a.status}; only active files can be moved to trash")
        rec = _move(layout, a, layout.trash, "trashed")
        db.flush()
        closed = _auto_resolve(db, ws, "")
        act = Action(workspace_id=ws, finding_id=None, action="DELETE", actor=actor,
                     affected=[rec] + ([{"auto_resolved": closed}] if closed else []),
                     note=f"Moved {a.path} to trash" + (f"; {len(closed)} related findings closed" if closed else ""))
        db.add(act)
        db.flush()
        audit(db, ws, actor, "asset.trashed", "asset", a.id, {"path": a.path, "action_id": act.id, "bytes": a.size_bytes})
        acts.append(act)
    record_snapshot(db, ws, get_settings(), f"Moved {len(acts)} file{'s' if len(acts) != 1 else ''} to trash")
    return acts


def restore_asset(db: Session, ws: str, a: Asset, actor: str) -> Action:
    """Bring a trashed file back by undoing the action that trashed it (a manual delete or an applied DELETE)."""
    if a.status != "trashed":
        raise ActionError(f"{a.path} is not in the trash")
    for act in db.scalars(select(Action).where(Action.workspace_id == ws, Action.status == "applied")
                          .order_by(Action.created_at.desc())):
        if any(r.get("asset_id") == a.id and r.get("new_status") == "trashed" for r in act.affected):
            others = [r["name"] for r in act.affected if r.get("asset_id") not in (None, a.id)]
            if others:
                raise ActionError(f"{a.path} was deleted together with {len(others)} other file(s); "
                                  "undo that action from Audit & Undo to restore them all")
            return undo_action(db, act, actor)
    raise ActionError(f"no undoable action found for {a.path}")
