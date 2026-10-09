"""Per-workspace storage layout and safe file handling. Paths never come from the client unchecked."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from app.config import get_settings

MAX_UPLOAD_BYTES = 2 * 1024**3
SKIP_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
SKIP_DIRS = {"__MACOSX", ".git"}


class Layout:
    def __init__(self, workspace_id: str):
        self.root = get_settings().workspaces_dir / workspace_id
        self.files = self.root / "files"            # scanned content
        self.connectors = self.root / "connectors"  # inventory exports, query logs, access manifest
        self.archive = self.root / "_archive"
        self.trash = self.root / "_trash"           # soft delete; restorable via undo
        self.secured = self.root / "_secured"       # owner-only vault (chmod 600)
        self.staging = self.root / "staging"        # dry-run outputs
        self.outputs = self.root / "outputs"        # committed pipeline outputs
        self.vault = self.root / ".dataguardian" / "vault"  # compressed originals of in-place commits
        self.journal = self.root / ".dataguardian" / "journal.jsonl"

    def ensure(self) -> "Layout":
        for p in (self.files, self.connectors, self.archive, self.trash, self.secured, self.staging, self.outputs, self.vault):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def inside(self, base: Path, rel: str) -> Path:
        target = (base / rel).resolve()
        if not str(target).startswith(str(base.resolve()) + "/") and target != base.resolve():
            raise ValueError(f"path escapes workspace: {rel}")
        return target


def asset_path(layout: Layout, a) -> Path | None:
    """Where an asset's bytes live right now (None for metadata-only connectors)."""
    if a.source == "pipeline-output":
        return (layout.trash if a.status == "trashed" else layout.outputs) / a.path
    if a.source != "upload":
        return None
    if (a.meta or {}).get("location") == "secured":
        return layout.secured / a.path
    if a.status == "archived":
        return layout.archive / a.path
    if a.status == "trashed":
        return layout.trash / a.path
    return layout.files / a.path


def safe_extract_zip(zip_path: Path, dest: Path) -> list[str]:
    """Extract with zip-slip, symlink and size protection. Returns extracted relative paths."""
    out: list[str] = []
    total = 0
    dest_res = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = info.filename
            parts = Path(name).parts
            if info.is_dir() or any(p in SKIP_DIRS for p in parts) or Path(name).name in SKIP_NAMES:
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:  # symlink
                continue
            total += info.file_size
            if total > MAX_UPLOAD_BYTES:
                raise ValueError("archive too large")
            target = (dest / name).resolve()
            if not str(target).startswith(str(dest_res) + "/"):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append(str(target.relative_to(dest_res)))
    return out


def safe_name(name: str) -> str:
    n = Path(name).name.replace("\x00", "")
    if not n or n in (".", ".."):
        raise ValueError("invalid file name")
    return n
