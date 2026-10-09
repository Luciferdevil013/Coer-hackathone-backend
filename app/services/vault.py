"""Atomic in-place commit of a Data-Patch, and rollback from the vault.

Commit (the "commit phase" of the virtual data tree):
  1. refuse a stale patch: the file must still hash to the fingerprint taken when the tree was mounted
  2. snapshot the original bytes into .dataguardian/vault/ (Zstandard; gzip if zstandard is missing),
     then decompress the snapshot and verify its SHA-256 before going any further
  3. copy the staged candidate next to the target, fsync, and os.replace() it over the target (atomic on POSIX)
  4. append to the rollback journal (.dataguardian/journal.jsonl) and the audit trail

Rollback reverses it: verify the file is still the committed version, restore the verified original
with the same atomic swap, restore the asset's metadata and reopen the findings the commit resolved.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.engine.health import record_snapshot
from app.models import Asset, Finding, PipelineRun, VaultCommit, new_id, utcnow
from app.pipeline import virtual as V
from app.scanner.readers import file_hash
from app.services.actions import ActionError, audit
from app.services.pipelines import resolve_findings
from app.services.storage import Layout, asset_path

try:
    import zstandard
except ImportError:  # pragma: no cover - requirements pin it; gzip keeps the vault working without it
    zstandard = None

CODEC = "zstd" if zstandard else "gzip"


def _compress(src: Path, dst: Path) -> None:
    with src.open("rb") as fi, dst.open("wb") as fo:
        if zstandard:
            zstandard.ZstdCompressor(level=10, threads=-1).copy_stream(fi, fo)
        else:
            with gzip.GzipFile(fileobj=fo, mode="wb", compresslevel=6) as gz:
                shutil.copyfileobj(fi, gz)
        fo.flush()
        os.fsync(fo.fileno())


def _decompress_to(src: Path, dst: Path, codec: str) -> str:
    """Decompress a vault file to `dst` and return the SHA-256 of what came out."""
    h = hashlib.sha256()
    with src.open("rb") as fi, dst.open("wb") as fo:
        reader = zstandard.ZstdDecompressor().stream_reader(fi) if codec == "zstd" else gzip.GzipFile(fileobj=fi)
        with reader:
            while chunk := reader.read(1 << 20):
                h.update(chunk)
                fo.write(chunk)
        fo.flush()
        os.fsync(fo.fileno())
    return h.hexdigest()


def _atomic_replace(target: Path, new_bytes_from: Path) -> None:
    tmp = target.with_name(f".{target.name}.dg-{new_id()}.tmp")  # same directory -> same filesystem -> atomic rename
    shutil.copyfile(new_bytes_from, tmp)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, target)


def _journal(layout: Layout, entry: dict) -> None:
    layout.journal.parent.mkdir(parents=True, exist_ok=True)
    with layout.journal.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": utcnow().isoformat() + "Z", **entry}) + "\n")


def commit_in_place(db: Session, run: PipelineRun, approver: str) -> VaultCommit:
    if run.status != "dry_run":
        raise ActionError(f"run is {run.status}; only a reviewed dry run can be committed")
    vdt = (run.metrics or {}).get("vdt") or {}
    if not vdt.get("in_place", {}).get("supported"):
        raise ActionError("in-place commit is not available for this file: "
                          + (vdt.get("in_place", {}).get("reason") or "dry-run it again first"))
    asset = db.get(Asset, run.asset_id)
    layout = Layout(run.workspace_id).ensure()
    target = asset_path(layout, asset)
    if asset.status != "active" or target is None or not target.exists():
        raise ActionError("the source file is no longer in place (archived, secured or deleted)")
    if V.sha256_file(target) != vdt["source_sha256"]:
        raise ActionError("stale patch: the file changed after this dry run. Run the dry run again before committing")
    candidate = layout.staging / run.id / "in_place" / asset.name
    if not candidate.exists():
        raise ActionError("the staged in-place candidate is missing; run the dry run again")

    commit_id = new_id()
    vault_name = f"{commit_id}__{asset.name}.{'zst' if CODEC == 'zstd' else 'gz'}"
    vault_file = layout.vault / vault_name
    _compress(target, vault_file)
    check = layout.vault / f".{commit_id}.verify"
    try:
        if _decompress_to(vault_file, check, CODEC) != vdt["source_sha256"]:
            vault_file.unlink(missing_ok=True)
            raise ActionError("vault snapshot failed verification; nothing was changed")
    finally:
        check.unlink(missing_ok=True)

    before = {"size_bytes": asset.size_bytes, "content_hash": asset.content_hash,
              "modified_at": asset.modified_at.isoformat() if asset.modified_at else None,
              "rows": (asset.meta or {}).get("rows")}
    _atomic_replace(target, candidate)  # <- the only write to the real file

    stem = asset.name.rsplit(".", 1)[0]
    qname, n = f"{stem}.quarantine.csv", 1
    while (layout.outputs / qname).exists():
        n += 1
        qname = f"{stem}.quarantine.v{n}.csv"
    shutil.copy2(run.quarantine_path, layout.outputs / qname)

    new_sha, new_bytes = V.sha256_file(target), target.stat().st_size
    asset.size_bytes, asset.content_hash, asset.modified_at = new_bytes, file_hash(target), utcnow()
    asset.meta = {**(asset.meta or {}), "rows": run.metrics["totals"]["rows_out"],
                  "vault": {"commit_id": commit_id, "run_id": run.id}}
    resolved = resolve_findings(db, run, asset)
    vc = VaultCommit(id=commit_id, workspace_id=run.workspace_id, run_id=run.id, asset_id=asset.id, file_name=asset.name,
                     vault_file=vault_name, codec=CODEC, original_sha256=vdt["source_sha256"],
                     original_bytes=vdt["source_bytes"], vault_bytes=vault_file.stat().st_size,
                     new_sha256=new_sha, new_bytes=new_bytes, actor=approver,
                     journal={"asset_before": before, "quarantine_file": qname, "resolved_findings": resolved,
                              "mutations": vdt["virtual_mutations"]})
    db.add(vc)
    run.status, run.output_asset_id, run.approved_by, run.committed_at = "committed", asset.id, approver, utcnow()
    run.metrics = {**run.metrics, "commit": {"mode": "in_place", "commit_id": commit_id, "quarantine": qname}}
    _journal(layout, {"event": "commit", "commit_id": commit_id, "run_id": run.id, "file": asset.name,
                      "vault": vault_name, "codec": CODEC, "original_sha256": vc.original_sha256, "new_sha256": new_sha})
    audit(db, run.workspace_id, approver, "vault.commit", "vault_commit", commit_id,
          {"file": asset.name, "run_id": run.id, "codec": CODEC, "original_bytes": vc.original_bytes,
           "vault_bytes": vc.vault_bytes, "new_bytes": new_bytes, "resolved_findings": resolved,
           **run.metrics["totals"]})
    record_snapshot(db, run.workspace_id, get_settings(), f"Committed patch to {asset.name}")
    return vc


def rollback(db: Session, vc: VaultCommit, actor: str) -> VaultCommit:
    if vc.status != "committed":
        raise ActionError(f"commit {vc.id} is already {vc.status.replace('_', ' ')}")
    layout = Layout(vc.workspace_id)
    asset = db.get(Asset, vc.asset_id)
    target = asset_path(layout, asset) if asset else None
    if target is None or not target.exists():
        raise ActionError("the committed file is no longer in place (archived, secured or deleted); restore it first")
    if V.sha256_file(target) != vc.new_sha256:
        newer = db.scalars(select(VaultCommit).where(VaultCommit.asset_id == vc.asset_id, VaultCommit.status == "committed",
                                                     VaultCommit.created_at > vc.created_at)).first()
        raise ActionError(f"roll back the newer commit {newer.id} first" if newer
                          else "the file was modified after this commit; rolling back would discard those changes")
    vault_file = layout.vault / vc.vault_file
    if not vault_file.exists():
        raise ActionError("vault snapshot is missing")
    restored = target.with_name(f".{target.name}.dg-restore-{vc.id}.tmp")
    try:
        if _decompress_to(vault_file, restored, vc.codec) != vc.original_sha256:
            raise ActionError("vault snapshot failed verification; the file was left as it is")
        os.replace(restored, target)
    finally:
        restored.unlink(missing_ok=True)

    b = vc.journal.get("asset_before", {})
    asset.size_bytes, asset.content_hash = b.get("size_bytes", vc.original_bytes), file_hash(target)
    asset.modified_at = utcnow()
    meta = {**(asset.meta or {})}
    meta.pop("vault", None)
    if b.get("rows") is not None:
        meta["rows"] = b["rows"]
    asset.meta = meta
    reopened = []
    for fid in vc.journal.get("resolved_findings", []):
        f = db.get(Finding, fid)
        if f and f.status == "resolved" and f.resolution == "CLEAN":
            f.status, f.resolution = "open", ""
            reopened.append(fid)
    q = vc.journal.get("quarantine_file")
    if q:
        (layout.outputs / q).unlink(missing_ok=True)
    run = db.get(PipelineRun, vc.run_id)
    if run:
        run.status = "rolled_back"
    vc.status, vc.rolled_back_by, vc.rolled_back_at = "rolled_back", actor, utcnow()
    _journal(layout, {"event": "rollback", "commit_id": vc.id, "file": vc.file_name, "restored_sha256": vc.original_sha256})
    audit(db, vc.workspace_id, actor, "vault.rollback", "vault_commit", vc.id,
          {"file": vc.file_name, "restored_bytes": vc.original_bytes, "reopened_findings": reopened})
    record_snapshot(db, vc.workspace_id, get_settings(), f"Rolled back {vc.file_name}")
    return vc


def verify(vc: VaultCommit) -> dict:
    """Integrity check: does the vault snapshot still decompress to the original bytes?"""
    layout = Layout(vc.workspace_id)
    vault_file = layout.vault / vc.vault_file
    if not vault_file.exists():
        return {"ok": False, "detail": "vault snapshot is missing"}
    tmp = layout.vault / f".{vc.id}.verify"
    try:
        ok = _decompress_to(vault_file, tmp, vc.codec) == vc.original_sha256
    finally:
        tmp.unlink(missing_ok=True)
    return {"ok": ok, "detail": "snapshot decompresses to the original SHA-256" if ok else "snapshot is corrupted"}


def ser_commit(vc: VaultCommit) -> dict:
    return {
        "id": vc.id, "run_id": vc.run_id, "asset_id": vc.asset_id, "file_name": vc.file_name, "codec": vc.codec,
        "vault_file": f".dataguardian/vault/{vc.vault_file}", "original_sha256": vc.original_sha256,
        "original_bytes": vc.original_bytes, "vault_bytes": vc.vault_bytes, "new_sha256": vc.new_sha256,
        "new_bytes": vc.new_bytes, "compression_ratio": round(vc.original_bytes / vc.vault_bytes, 2) if vc.vault_bytes else None,
        "status": vc.status, "actor": vc.actor, "created_at": vc.created_at.isoformat(),
        "rolled_back_by": vc.rolled_back_by, "rolled_back_at": vc.rolled_back_at.isoformat() if vc.rolled_back_at else None,
        "mutations": vc.journal.get("mutations", {}), "quarantine_file": vc.journal.get("quarantine_file"),
    }

