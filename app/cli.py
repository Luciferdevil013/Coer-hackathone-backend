"""dataguardian CLI: inspect and roll back in-place commits without the dashboard.

    ./dataguardian commits [--workspace ID]
    ./dataguardian verify <commit_id>
    ./dataguardian rollback <commit_id> [--actor NAME]
"""
from __future__ import annotations

import argparse
import sys

from sqlalchemy import select

from app.db import session_scope
from app.models import VaultCommit, Workspace
from app.services import vault
from app.services.actions import ActionError


def _size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return str(n)


def cmd_commits(args) -> int:
    with session_scope() as db:
        q = select(VaultCommit).order_by(VaultCommit.created_at.desc())
        if args.workspace:
            q = q.where(VaultCommit.workspace_id == args.workspace)
        rows = db.scalars(q.limit(50)).all()
        names = {w.id: w.name for w in db.scalars(select(Workspace))}
        if not rows:
            print("No in-place commits.")
        for c in rows:
            print(f"{c.id}  {c.status:<11}  {c.created_at:%Y-%m-%d %H:%M}  {c.file_name:<32}  "
                  f"{_size(c.original_bytes)} -> vault {_size(c.vault_bytes)} ({c.codec})  [{names.get(c.workspace_id, c.workspace_id)}]")
    return 0


def _get(db, commit_id: str) -> VaultCommit:
    vc = db.get(VaultCommit, commit_id)
    if not vc:
        sys.exit(f"error: no commit {commit_id}")
    return vc


def cmd_verify(args) -> int:
    with session_scope() as db:
        r = vault.verify(_get(db, args.commit_id))
    print(("OK  " if r["ok"] else "FAIL  ") + r["detail"])
    return 0 if r["ok"] else 1


def cmd_rollback(args) -> int:
    with session_scope() as db:
        vc = _get(db, args.commit_id)
        try:
            vault.rollback(db, vc, args.actor)
        except ActionError as e:
            sys.exit(f"error: {e}")
        print(f"Rolled back {vc.file_name}: restored {_size(vc.original_bytes)} from the vault "
              f"(sha256 {vc.original_sha256[:12]}… verified).")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dataguardian", description="DataGuardian vault commits and rollback")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("commits", help="list in-place commits")
    p.add_argument("--workspace")
    p.set_defaults(fn=cmd_commits)
    p = sub.add_parser("verify", help="check a vault snapshot still decompresses to the original")
    p.add_argument("commit_id")
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("rollback", help="restore the original file from the vault")
    p.add_argument("commit_id")
    p.add_argument("--actor", default="cli")
    p.set_defaults(fn=cmd_rollback)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
