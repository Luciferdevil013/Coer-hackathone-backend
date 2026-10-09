"""Validate any PipelineSpec (AI-generated or hand-edited) before it can run."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

from pydantic import ValidationError

from app.pipeline.spec import CATALOG, PipelineSpec, Step
from app.scanner.profiler import ColumnProfile

MAX_STEPS = 20
_ORDER_AFTER = {"deduplicate": {"normalize_email", "normalize_phone", "trim_whitespace", "decode_html", "repair_text"},
                "quarantine_anomalies": {"trim_whitespace"},
                "validate_rules": {"normalize_phone", "normalize_email", "parse_dates"}}


@dataclass
class ValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    spec: PipelineSpec | None = None

    def to_dict(self) -> dict:
        return {"valid": self.valid, "errors": self.errors, "warnings": self.warnings}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def resolve_column(name: str, columns: list[str]) -> str | None:
    if name in columns:
        return name
    n = _norm(name)
    for c in columns:
        if _norm(c) == n:
            return c
    # RapidFuzz fallback ("phone_number" -> "phone"); only an unambiguous, strong best match
    if not columns or len(n) < 3:
        return None
    hits = process.extract(n, {c: _norm(c) for c in columns}, scorer=fuzz.WRatio, limit=2)
    if hits and hits[0][1] >= 90 and (len(hits) == 1 or hits[0][1] - hits[1][1] >= 8):
        return hits[0][2]
    return None


def validate_spec(spec_in: dict | PipelineSpec, columns: list[str], profiles: dict[str, ColumnProfile] | None = None) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        spec = spec_in if isinstance(spec_in, PipelineSpec) else PipelineSpec.model_validate(spec_in)
    except ValidationError as e:
        return ValidationReport(False, [f"Spec schema error: {err['loc']}: {err['msg']}" for err in e.errors()])
    if not spec.steps:
        errors.append("Pipeline has no steps")
    if len(spec.steps) > MAX_STEPS:
        errors.append(f"Too many steps ({len(spec.steps)} > {MAX_STEPS})")

    available = list(columns)
    fixed_steps: list[Step] = []
    seen_ops: list[str] = []
    for i, step in enumerate(spec.steps, start=1):
        label = f"Step {i} ({step.op})"
        info = CATALOG.get(step.op)
        if info is None:
            errors.append(f"{label}: unsupported operation — allowed: {', '.join(CATALOG)}")
            continue
        try:
            params = info.params_model.model_validate(step.params)
        except ValidationError as e:
            errors.append(f"{label}: invalid parameters — " + "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()))
            continue
        data = params.model_dump()

        def fix(cols: list[str], allow_star: bool = False) -> list[str]:
            out = []
            for c in cols:
                if allow_star and c == "*":
                    out.append(c)
                    continue
                r = resolve_column(c, available)
                if r is None:
                    errors.append(f"{label}: column '{c}' does not exist (available: {', '.join(available[:12])})")
                else:
                    if r != c:
                        warnings.append(f"{label}: matched column '{c}' to '{r}'")
                    out.append(r)
            return out

        for key in info.column_params:
            if key in data:
                data[key] = fix(data[key], allow_star=(key == "columns" and step.op in ("trim_whitespace", "normalize_email", "quarantine_anomalies", "decode_html", "redact_text_pii", "repair_text")))
        if step.op in ("validate_rules",):
            data["formats"] = {(resolve_column(k, available) or k): v for k, v in data["formats"].items()}
            data["allowed_values"] = {(resolve_column(k, available) or k): v for k, v in data["allowed_values"].items()}
            for k in list(data["formats"]) + list(data["allowed_values"]):
                if k not in available:
                    errors.append(f"{label}: column '{k}' does not exist")
            if not any([data["required"], data["formats"], data["allowed_values"], data["unique"]]):
                errors.append(f"{label}: no rules to check")
        if step.op in ("standardize_case", "normalize_phone", "parse_dates", "fill_missing", "mask_pii", "drop_columns") and not data.get("columns"):
            errors.append(f"{label}: no columns selected")

        if profiles:
            for c in data.get("columns", []):
                p = profiles.get(c)
                if not p:
                    continue
                if step.op == "normalize_phone" and p.pii_entity != "PHONE" and p.inferred_type != "phone":
                    warnings.append(f"{label}: '{c}' does not look like a phone column ({p.inferred_type})")
                if step.op == "parse_dates" and p.inferred_type != "date":
                    warnings.append(f"{label}: '{c}' does not look like a date column ({p.inferred_type})")
                if step.op == "normalize_email" and p.pii_entity != "EMAIL" and c != "*":
                    warnings.append(f"{label}: '{c}' does not look like an email column")
        if step.op in _ORDER_AFTER:
            later = [op for op in _ORDER_AFTER[step.op] if op in {s.op for s in spec.steps[i:]}]
            if later:
                warnings.append(f"{label}: runs before {', '.join(later)} — results may differ; consider moving it later")
        if info.destructive:
            warnings.append(f"{label}: removes rows/columns from the output — shown in dry run, requires approval to commit")
        if step.op == "drop_columns":
            available = [c for c in available if c not in data["columns"]]
        seen_ops.append(step.op)
        fixed_steps.append(Step(id=step.id or f"s{i}", op=step.op, params=data, reason=step.reason))

    ids = [s.id for s in fixed_steps]
    if len(ids) != len(set(ids)):
        errors.append("Step ids must be unique")
    spec = spec.model_copy(update={"steps": fixed_steps})
    return ValidationReport(not errors, errors, list(dict.fromkeys(warnings)), spec)


def graph(spec: PipelineSpec, source_name: str) -> dict:
    """Nodes/edges view of a (linear) pipeline for the visual editor."""
    nodes = [{"id": "src", "type": "source", "label": "Read source", "detail": source_name}]
    for s in spec.steps:
        info = CATALOG.get(s.op)
        nodes.append({"id": s.id, "type": "op", "op": s.op, "label": info.title if info else s.op,
                      "params": s.params, "reason": s.reason,
                      "destructive": bool(info and info.destructive), "lossy": bool(info and info.lossy)})
    nodes.append({"id": "out", "type": "sink", "label": "Write to staging", "detail": "new file · original untouched"})
    edges = [{"from": nodes[i]["id"], "to": nodes[i + 1]["id"]} for i in range(len(nodes) - 1)]
    return {"nodes": nodes, "edges": edges}
