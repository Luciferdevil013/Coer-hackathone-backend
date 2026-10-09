"""Clean Studio service: plan -> validate -> dry run (virtual data tree + staging) -> approve -> commit.

A commit either writes a new file (default) or applies the Data-Patch in place through the vault
(services/vault.py), which keeps the compressed original for rollback.
"""
from __future__ import annotations

import json
import shutil

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.engine.health import latest_scan, record_snapshot
from app.models import Asset, Finding, PipelineRun, RuleSet, utcnow
from app.pipeline import executor
from app.pipeline import virtual as V
from app.pipeline.planner import plan as make_plan
from app.pipeline.planner import rules_plan
from app.pipeline.spec import PipelineSpec, Step
from app.scanner.readers import effective_ext
from app.pipeline.validator import graph, validate_spec
from app.scanner import readers
from app.scanner.profiler import DatasetView, profile_dataframe
from app.services.actions import ActionError, audit
from app.services.storage import Layout, asset_path


def active_rules(db: Session, ws: str) -> tuple[dict, int]:
    rs = db.scalars(select(RuleSet).where(RuleSet.workspace_id == ws, RuleSet.active.is_(True))
                    .order_by(RuleSet.version.desc())).first()
    if not rs:
        return {}, 0
    return {**rs.rules, "_version": rs.version}, rs.version


def load_view(db: Session, asset: Asset) -> DatasetView:
    reason = (asset.meta or {}).get("read_error")
    if asset.kind != "tabular":
        raise ActionError(f"{asset.name} was not scanned: {reason}" if reason
                          else "Clean Studio works on tabular files (CSV, Excel, JSON records)")
    path = asset_path(Layout(asset.workspace_id), asset)
    if path is None or not path.exists():
        raise ActionError("file content is not available")
    s = get_settings()
    errs: list[str] = []
    df = readers.read_table(path, s.max_rows_per_dataset, errs)
    if df is None:
        raise ActionError(f"{asset.name} could not be read as a table: {errs[0] if errs else reason or 'no columns found'}")
    v = DatasetView(asset.id, asset.name, df)
    profile_dataframe(v, s.pii_sample_values, s.pii_column_match_ratio)
    return v


def review_merge_step(db: Session, ws: str, asset_id: str) -> tuple[Step | None, int]:
    """Build a merge_rows step from the human decisions on the latest duplicate-rows finding."""
    scan = latest_scan(db, ws)
    if not scan:
        return None, 0
    f = db.scalars(select(Finding).where(Finding.workspace_id == ws, Finding.scan_id == scan.id,
                                         Finding.kind == "duplicate_rows")).all()
    f = next((x for x in f if asset_id in x.target_asset_ids), None)
    if not f:
        return None, 0
    clusters = [c["row_ids"] for c in f.details.get("clusters", []) if c.get("decision") == "merge"]
    if not clusters:
        return None, 0
    conflict = f.details.get("keys", {}).get("verified", [])
    return Step(id="review", op="merge_rows", params={"clusters": clusters, "keep": "most_complete",
                                                      "conflict_columns": conflict},
                reason=f"{len(clusters)} duplicate groups approved in review"), len(clusters)


def plan_pipeline(db: Session, ws: str, asset: Asset, prompt: str, use_review: bool = True) -> dict:
    view = load_view(db, asset)
    rules, version = active_rules(db, ws)
    draft = make_plan(prompt, view, rules, ws)
    steps = list(draft.steps)
    reviewed = 0
    if use_review:
        merge, reviewed = review_merge_step(db, ws, asset.id)
        if merge:
            # human-approved merges run before automatic dedupe so reviewers' decisions take precedence
            pos = next((i for i, s in enumerate(steps) if s.op in ("deduplicate", "validate_rules", "mask_pii", "drop_columns")), len(steps))
            steps.insert(pos, merge)
    for i, s in enumerate(steps, 1):
        s.id = f"s{i}"
    spec = PipelineSpec(source_asset_id=asset.id, steps=steps)
    report = validate_spec(spec, list(view.df.columns), view.profiles)
    note = draft.note
    if draft.planner == "llm" and not report.valid:  # never run an invalid AI plan; fall back
        fb = rules_plan(prompt, view, rules)
        fb_spec = PipelineSpec(source_asset_id=asset.id, steps=fb.steps)
        fb_report = validate_spec(fb_spec, list(view.df.columns), view.profiles)
        if fb_report.valid:
            note = "The AI plan failed validation (" + "; ".join(report.errors[:2]) + "), so the deterministic planner was used."
            draft, report = fb, fb_report
    final = report.spec or spec
    return {
        "spec": final.model_dump(),
        "graph": graph(final, asset.name),
        "validation": report.to_dict(),
        "assumptions": draft.assumptions,
        "questions": draft.questions,
        "unsupported": draft.unsupported,
        "planner": draft.planner,
        "note": note,
        "rule_set_version": version,
        "reviewed_groups": reviewed,
        "profile": [p.to_dict() for p in view.profiles.values()],
        "rows": view.rows,
    }


def dry_run(db: Session, ws: str, asset: Asset, spec_in: dict, prompt: str, planner: str, actor: str) -> PipelineRun:
    view = load_view(db, asset)
    spec_in = {**spec_in, "source_asset_id": asset.id}  # never trust the client/model for this
    report = validate_spec(spec_in, list(view.df.columns), view.profiles)
    if not report.valid:
        raise ActionError("Pipeline is invalid: " + "; ".join(report.errors))
    _, version = active_rules(db, ws)
    result = executor.execute(view.df, report.spec)
    layout = Layout(ws).ensure()
    run = PipelineRun(workspace_id=ws, asset_id=asset.id, prompt=prompt, spec=report.spec.model_dump(),
                      planner=planner, rule_set_version=version, status="dry_run")
    db.add(run)
    db.flush()
    stage_dir = layout.staging / run.id
    stage_dir.mkdir(parents=True, exist_ok=True)
    stem = asset.name.rsplit(".", 1)[0]
    out_path, q_path = stage_dir / f"{stem}.cleaned.csv", stage_dir / f"{stem}.quarantine.csv"
    executor.to_csv(result.output, out_path)
    executor.to_csv(result.quarantine, q_path, keep_reason=True)
    sensitive = {c: p.pii_entity for c, p in view.profiles.items() if p.pii_entity}
    vdt = mount_virtual_tree(run, asset, view, result, sensitive, stage_dir)
    run.metrics = {"totals": result.totals, "steps": result.steps, "warnings": report.warnings, "vdt": vdt}
    run.preview = executor.preview(result, sensitive)
    run.staging_path, run.quarantine_path = str(out_path), str(q_path)
    audit(db, ws, actor, "pipeline.dry_run", "pipeline_run", run.id,
          {"asset": asset.name, "planner": planner, "rule_set_version": version, **result.totals,
           "source_sha256": vdt["source_sha256"], "patch_entries": vdt["patch_entries"]})
    return run


def mount_virtual_tree(run: PipelineRun, asset: Asset, view: DatasetView, result: executor.ExecutionResult,
                       sensitive: dict, stage_dir) -> dict:
    """Reconcile the shadow against the source, stage the in-place candidate and the Data-Patch."""
    path = asset_path(Layout(asset.workspace_id), asset)
    ext = effective_ext(asset.name)
    tree = V.build(run.id, asset.name, view.df, result, sensitive)
    V.remember(tree)
    ok, why = V.in_place_support(path, ext, view.rows, get_settings().max_rows_per_dataset, list(view.df.columns))
    source_bytes = path.stat().st_size
    candidate_bytes = None
    if ok:
        cand = stage_dir / "in_place" / asset.name
        cand.parent.mkdir(parents=True, exist_ok=True)
        V.write_like_source(result.output, cand, ext)
        candidate_bytes = cand.stat().st_size
    vdt = {
        "engine": V.ENGINE, "target_file": asset.name, "source_sha256": V.sha256_file(path),
        "source_bytes": source_bytes, "candidate_bytes": candidate_bytes, "total_records": view.rows,
        "in_place": {"supported": ok, "reason": why, "format": ext},
        "virtual_mutations": V.mutations(tree, source_bytes, candidate_bytes),
        "patch_entries": len(tree.patch),
    }
    meta = {k: vdt[k] for k in ("engine", "target_file", "source_sha256", "total_records", "virtual_mutations")}
    (stage_dir / "data-patch.json").write_text(json.dumps({"run_id": run.id, **V.masked_patch(tree, meta)},
                                                          ensure_ascii=False, indent=1), encoding="utf-8")
    return vdt


def mount_run(db: Session, run: PipelineRun) -> V.VirtualTree:
    """The mounted tree for a run; re-mounted from the source if this process no longer holds it."""
    tree = V.cached(run.id)
    if tree:
        return tree
    vdt = (run.metrics or {}).get("vdt")
    if run.status != "dry_run" or not vdt:
        raise ActionError("this run's virtual tree is no longer mounted; start a new dry run to inspect it")
    asset = db.get(Asset, run.asset_id)
    view = load_view(db, asset)
    if V.sha256_file(asset_path(Layout(asset.workspace_id), asset)) != vdt["source_sha256"]:
        raise ActionError("the source file changed since this dry run; the patch is stale, so run it again")
    result = executor.execute(view.df, PipelineSpec.model_validate(run.spec))
    sensitive = {c: p.pii_entity for c, p in view.profiles.items() if p.pii_entity}
    tree = V.build(run.id, asset.name, view.df, result, sensitive)
    V.remember(tree)
    return tree


def discard_run(db: Session, run: PipelineRun, actor: str) -> PipelineRun:
    """Reject: throw the virtual patch away. Nothing was ever written outside staging."""
    if run.status != "dry_run":
        raise ActionError(f"run is {run.status}; only a dry run can be discarded")
    if run.staging_path:
        shutil.rmtree(Layout(run.workspace_id).staging / run.id, ignore_errors=True)
    V.unmount(run.id)
    run.status = "discarded"
    audit(db, run.workspace_id, actor, "pipeline.discard", "pipeline_run", run.id,
          {"patch_entries": (run.metrics.get("vdt") or {}).get("patch_entries", 0)})
    return run


FIXES = {
    "duplicate_rows": {"deduplicate", "merge_rows"},
    "anomalous_values": {"quarantine_anomalies"},
    "data_quality": {"trim_whitespace", "normalize_email", "normalize_phone", "parse_dates", "validate_rules", "standardize_case",
                     "fill_missing", "decode_html", "drop_columns", "repair_text"},
    "pii_in_text": {"redact_text_pii"},
}


def _addresses(kind: str, ops: set[str]) -> bool:
    """A cleaned output only resolves the findings its pipeline actually fixed."""
    return bool(FIXES.get(kind, set()) & ops)


def resolve_findings(db: Session, run: PipelineRun, src: Asset) -> list[str]:
    scan = latest_scan(db, run.workspace_id)
    ops = {s["op"] for s in run.spec["steps"]}
    resolved = []
    if scan:
        for f in db.scalars(select(Finding).where(Finding.workspace_id == run.workspace_id, Finding.scan_id == scan.id,
                                                  Finding.status == "open", Finding.action == "CLEAN")):
            if src.id in f.target_asset_ids and _addresses(f.kind, ops):
                f.status, f.resolution = "resolved", "CLEAN"
                resolved.append(f.id)
    return resolved


def commit_run(db: Session, run: PipelineRun, approver: str) -> PipelineRun:
    if run.status != "dry_run":
        raise ActionError(f"run is {run.status}; only a reviewed dry run can be committed")
    layout = Layout(run.workspace_id).ensure()
    src = db.get(Asset, run.asset_id)
    stem = src.name.rsplit(".", 1)[0]
    name, n = f"{stem}.cleaned.csv", 1
    while (layout.outputs / name).exists():
        n += 1
        name = f"{stem}.cleaned.v{n}.csv"
    shutil.copy2(run.staging_path, layout.outputs / name)
    qname = name.replace(".cleaned", ".quarantine")
    shutil.copy2(run.quarantine_path, layout.outputs / qname)
    out = Asset(workspace_id=run.workspace_id, source="pipeline-output", path=name, name=name, ext="csv",
                kind="tabular", size_bytes=(layout.outputs / name).stat().st_size, modified_at=utcnow(),
                accessed_at=utcnow(), owner=src.owner, department=src.department, shared_with=src.shared_with,
                encrypted=src.encrypted, content_hash="", content_available=True,
                meta={"lineage": {"source_asset_id": src.id, "source_name": src.name, "run_id": run.id,
                                  "rule_set_version": run.rule_set_version, "planner": run.planner,
                                  "operators": [f"{s['op']}@1.0" for s in run.spec["steps"]]},
                      "rows": run.metrics["totals"]["rows_out"], "quarantine_file": qname})
    db.add(out)
    db.flush()
    run.status, run.output_asset_id, run.approved_by, run.committed_at = "committed", out.id, approver, utcnow()
    run.metrics = {**run.metrics, "commit": {"mode": "new_file", "output": name}}
    resolved = resolve_findings(db, run, src)
    audit(db, run.workspace_id, approver, "pipeline.commit", "pipeline_run", run.id,
          {"output": name, "quarantine": qname, "resolved_findings": resolved, **run.metrics["totals"]})
    record_snapshot(db, run.workspace_id, get_settings(), f"Cleaned {src.name}")
    return run
