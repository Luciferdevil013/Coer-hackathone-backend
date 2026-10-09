"""Connectors -> Asset rows.

* Local folder connector: real files, content scanned.
* S3 Inventory connector: metadata-only objects (an S3 Inventory / Drive export CSV).
* Access manifest: ownership / sharing / last-access metadata a file system can't provide.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Asset
from app.scanner.readers import detect_kind, effective_ext, file_hash
from app.services.storage import SKIP_DIRS, SKIP_NAMES, Layout


def _parse_dt(v: str | None) -> datetime | None:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "")).replace(tzinfo=None)
    except ValueError:
        return None


def load_manifest(layout: Layout) -> dict:
    p = layout.connectors / "access_manifest.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def sync_local(db: Session, workspace_id: str, layout: Layout) -> tuple[int, int]:
    manifest = load_manifest(layout)
    entries = manifest.get("files", {})
    defaults = manifest.get("defaults", {})
    existing = {
        a.path: a for a in db.scalars(select(Asset).where(Asset.workspace_id == workspace_id, Asset.source == "upload"))
    }
    seen, created = set(), 0
    for root, dirs, files in os.walk(layout.files):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".git")]
        for fn in files:
            if fn in SKIP_NAMES:
                continue
            p = Path(root) / fn
            rel = str(p.relative_to(layout.files))
            st = p.stat()
            mtime = datetime.fromtimestamp(st.st_mtime)
            meta_entry = {**defaults, **entries.get(rel, {})}
            a = existing.get(rel)
            if a is None:
                a = Asset(workspace_id=workspace_id, source="upload", path=rel, name=fn, meta={})
                db.add(a)
                created += 1
            changed = a.size_bytes != st.st_size or (a.meta or {}).get("fs_mtime") != st.st_mtime or not a.content_hash
            a.meta = {**(a.meta or {}), "fs_mtime": st.st_mtime}
            a.name, a.ext, a.kind = fn, effective_ext(fn), detect_kind(fn)
            a.size_bytes, a.content_available = st.st_size, True
            a.modified_at = _parse_dt(meta_entry.get("modified_at")) or mtime
            a.accessed_at = _parse_dt(meta_entry.get("last_accessed")) or a.modified_at
            a.owner = meta_entry.get("owner", a.owner or "unknown")
            a.department = meta_entry.get("department", a.department or "")
            a.shared_with = int(meta_entry.get("shared_with", a.shared_with or 1))
            a.encrypted = bool(meta_entry.get("encrypted", a.encrypted or False))
            if a.status in ("missing",):
                a.status = "active"
            if changed:
                a.content_hash = file_hash(p)
            seen.add(rel)
    for rel, a in existing.items():
        if rel not in seen and a.status == "active" and (a.meta or {}).get("location") != "secured":
            a.status = "missing"  # removed outside DataGuardian
    db.flush()
    return len(seen), created


INVENTORY_FIELDS = ("bucket", "key", "size", "last_modified", "etag", "storage_class")


def sync_inventory(db: Session, workspace_id: str, layout: Layout) -> int:
    p = layout.connectors / "s3_inventory.csv"
    if not p.exists():
        return 0
    existing = {
        a.path: a for a in db.scalars(select(Asset).where(Asset.workspace_id == workspace_id, Asset.source == "s3-inventory"))
    }
    n = 0
    with p.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not all(k in r for k in INVENTORY_FIELDS):
                raise ValueError(f"inventory CSV must have columns: {', '.join(INVENTORY_FIELDS)}")
            path = f"{r['bucket']}/{r['key']}"
            a = existing.get(path)
            if a is None:
                a = Asset(workspace_id=workspace_id, source="s3-inventory", path=path, meta={})
                db.add(a)
                existing[path] = a
            name = r["key"].rsplit("/", 1)[-1]
            a.name, a.ext, a.kind = name, effective_ext(name), detect_kind(name)
            a.size_bytes = int(float(r["size"] or 0))
            a.modified_at = _parse_dt(r["last_modified"])
            a.accessed_at = _parse_dt(r.get("last_accessed")) or a.modified_at
            a.content_hash = (r.get("etag") or "").strip('"')
            a.content_available = False
            if a.status == "active":  # our own lifecycle actions are the source of truth
                a.storage_class = r.get("storage_class") or "STANDARD"
            a.owner = r.get("owner") or "unknown"
            a.department = r.get("department") or ""
            a.shared_with = int(r.get("shared_with") or 1)
            a.encrypted = (r.get("encrypted") or "").lower() in ("true", "1", "yes", "sse-s3", "sse-kms")
            n += 1
    db.flush()
    return n
