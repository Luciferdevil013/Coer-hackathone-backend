"""Sensitive-data and secret findings: PII inventory + exposure + DPDP retention."""
from __future__ import annotations

import math

from app.engine.signals import ScanContext, Signal, ev, fmt_bytes
from app.models import Asset
from app.scanner import pii


def _risk(sensitivity: float, shared_with: int, encrypted: bool, records: int) -> float:
    exposure = min(1.0, shared_with / 50) + (0.0 if encrypted else 0.2)
    volume = min(1.0, math.log10(records + 1) / 6)
    return round(100 * (0.5 * sensitivity + 0.3 * min(exposure, 1.0) + 0.2 * volume), 1)


def _entity_summary(counts: dict[str, int]) -> str:
    return ", ".join(f"{pii.ENTITIES[k].label} ({v:,})" for k, v in sorted(counts.items(), key=lambda x: -pii.ENTITIES[x[0]].sensitivity))


def _records_with_pii(view, cols: list[str]) -> int:
    if not cols:
        return 0
    sub = view.df[cols]
    mask = sub.notna() & (sub.astype(str).apply(lambda s: s.str.strip()) != "")
    return int(mask.any(axis=1).sum())


def detect_sensitive(ctx: ScanContext) -> list[Signal]:
    s = ctx.settings
    signals: list[Signal] = []
    for a in ctx.active():
        entity_counts: dict[str, int] = {}
        columns: dict[str, str] = {}
        records = 0
        if a.id in ctx.views:
            v = ctx.views[a.id]
            columns = v.pii_columns()
            for col, ent in columns.items():
                entity_counts[ent] = entity_counts.get(ent, 0) + v.profiles[col].non_null
            records = _records_with_pii(v, list(columns))
        elif a.id in ctx.db_views:
            for tv in ctx.db_views[a.id]:
                tcols = tv.pii_columns()
                for col, ent in tcols.items():
                    columns[f"{tv.label}.{col}"] = ent
                    entity_counts[ent] = entity_counts.get(ent, 0) + tv.profiles[col].non_null
                records += _records_with_pii(tv, list(tcols))
        elif a.id in ctx.texts:
            scan = pii.scan_text(ctx.texts[a.id])
            if scan.secrets:
                signals.append(_secret_signal(ctx, a, dict(scan.secrets)))
            entity_counts = {k: v for k, v in scan.entities.items() if k != "EMAIL" or v >= 3}
            records = max(entity_counts.values()) if entity_counts else 0
        if not entity_counts:
            continue
        sens = max(pii.ENTITIES[e].sensitivity for e in entity_counts)
        days = ctx.days_since_use(a)
        past_retention = days > s.retention_days
        # Contact-only data (email/phone/name) is everywhere; flag it only when over-exposed or kept past retention.
        # Race, religion and similar fields (0.8) are sensitive on their own, like health or ID numbers.
        if sens < 0.8 and not (a.shared_with > s.exposure_users or past_retention):
            continue
        if sens < 0.5 and records < 100:
            continue
        risk = _risk(sens, a.shared_with, a.encrypted, records)
        high = [e for e in entity_counts if pii.ENTITIES[e].sensitivity >= 0.9]
        exposed = a.shared_with > s.exposure_users
        if past_retention:
            action, conf = "DELETE", 0.72
            summary = (f"{records:,} records with personal data, unused for {days:,} days (retention {s.retention_days} days). "
                       "Under the DPDP Act, data must be erased once its purpose is served — an erasure candidate after notice.")
        else:
            action, conf = "SECURE", 0.9 if exposed or not a.encrypted else 0.75
            summary = (f"{records:,} records with {', '.join(pii.ENTITIES[e].label for e in (high or list(entity_counts)))} "
                       f"accessible to {a.shared_with} users" + ("" if a.encrypted else ", stored unencrypted") + ".")
        title_ent = pii.ENTITIES[max(entity_counts, key=lambda e: pii.ENTITIES[e].sensitivity)].label
        signals.append(Signal(
            kind="sensitive_data", category="risk",
            title=f"{title_ent} data in {a.name} ({records:,} record{'s' if records != 1 else ''})",
            summary=summary,
            target_ids=[a.id],
            evidence=[
                ev("PII detected", _entity_summary(entity_counts)),
                ev("Records containing PII", f"{records:,}"),
                ev("Users with access", a.shared_with),
                ev("Encrypted at rest", "Yes" if a.encrypted else "No"),
                ev("Owner", f"{a.owner}" + (f" ({a.department})" if a.department else "")),
                ev("Last used", f"{days:,} days ago"),
                ev("Detection", "regex + checksum validation (Verhoeff/Luhn), locally — no data sent to AI"),
            ],
            details={"entities": entity_counts, "columns": columns, "past_retention": past_retention,
                     "days_since_use": days, "dpdp_relevant": True},
            confidence=conf, action=action, records=records, risk=risk,
        ))
    return signals


def detect_text_pii(ctx: ScanContext) -> list[Signal]:
    """Personal data pasted *inside* free-text cells (a tweet with an email, a note with a mobile number).
    Column-level detection misses it because the column as a whole is prose, not an email/phone field."""
    signals: list[Signal] = []
    views = [(a, ctx.views[a.id], None) for a in ctx.active() if a.id in ctx.views]
    views += [(a, tv, tv.label) for a in ctx.active() if a.id in ctx.db_views for tv in ctx.db_views[a.id]]
    for a, v, table in views:
        cols = {c: p for c, p in v.profiles.items() if p.text_pii}
        if not cols:
            continue
        ents: dict[str, int] = {}
        for p in cols.values():
            for k, n in p.text_pii.items():
                ents[k] = ents.get(k, 0) + n
        rows_hit = len(set().union(*(p.text_pii_row_ids for p in cols.values()))) or sum(p.text_pii_rows for p in cols.values())
        sens = max(pii.ENTITIES[e].sensitivity for e in ents)
        risk = round(_risk(sens, a.shared_with, a.encrypted, rows_hit) * 0.85, 1)  # scattered, not a whole column
        where = ", ".join(f"'{c}'" for c in cols)
        name = f"{a.name}" + (f" › {table}" if table else "")
        examples = [pii.mask_text(str(x))[:160] for c in cols for x in v.df[c].dropna().tolist()
                    if pii.scan_text(str(x), secrets=False).entities][:5]
        signals.append(Signal(
            kind="pii_in_text", category="risk",
            title=f"Personal data hidden in free text: {_entity_summary(ents)} in {name}",
            summary=(f"{rows_hit:,} row{'s' if rows_hit != 1 else ''} of {where} contain contact or ID details typed into the text. "
                     "They are easy to miss because the column is not a contact field. Redact them in a cleaned copy before sharing or analysis."),
            target_ids=[a.id],
            evidence=[
                ev("Found inside", where + (f" (table {table})" if table else "")),
                ev("PII detected", _entity_summary(ents)),
                ev("Rows affected", f"{rows_hit:,} of {v.rows:,}"),
                ev("Users with access", a.shared_with),
                ev("Detection", "regex + checksum validation on every cell, locally — no data sent to AI"),
            ],
            details={"entities": ents, "columns": {c: "free text" for c in cols}, "examples": examples, "table": table or "",
                     "dpdp_relevant": True,
                     "suggested_prompt": "Redact personal data in " + " and ".join(f"'{c}'" for c in cols)},
            confidence=0.9, action="CLEAN", records=rows_hit, risk=risk,
        ))
    return signals


def _secret_signal(ctx: ScanContext, a: Asset, secrets: dict[str, int]) -> Signal:
    total = sum(secrets.values())
    return Signal(
        kind="secret_exposed", category="risk",
        title=f"{total} credential{'s' if total > 1 else ''} exposed in {a.name}",
        summary=f"Plain-text secrets ({', '.join(secrets)}) found in a file shared with {a.shared_with} users. Rotate them and restrict the file.",
        target_ids=[a.id],
        evidence=[
            ev("Secret types", ", ".join(f"{k} ×{v}" for k, v in secrets.items())),
            ev("File", a.path),
            ev("Users with access", a.shared_with),
            ev("Values shown", "never — secrets are redacted before display or AI use"),
        ],
        details={"secrets": secrets},
        confidence=0.95, action="SECURE", records=total, risk=95.0,
    )


def detect_stale_large(ctx: ScanContext) -> list[Signal]:
    s = ctx.settings
    cands = []
    for a in ctx.active(include_outputs=False):
        if a.id in ctx.claimed:
            continue
        threshold = s.large_object_bytes if a.source == "s3-inventory" else s.large_file_bytes
        days = ctx.days_since_use(a)
        if a.size_bytes >= threshold and days >= s.stale_days:
            cands.append((a, days))
    cands.sort(key=lambda x: -x[0].size_bytes)
    local = [c for c in cands if c[0].source != "s3-inventory"][:10]
    cloud = [c for c in cands if c[0].source == "s3-inventory"]
    individual, rest_pool = local + cloud[:5], cloud[5:]
    signals = []
    for a, days in individual:
        ctx.claimed.add(a.id)
        value = "LOW" if a.shared_with <= 2 and days > 2 * s.stale_days else "MEDIUM"
        zip_meta = a.meta.get("zip") or {}
        contents = ""
        if zip_meta:
            yrs = zip_meta.get("member_years") or []
            span = (f"{yrs[0]}" if yrs[0] == yrs[-1] else f"{yrs[0]}–{yrs[-1]}") if yrs else ""
            contents = f"{zip_meta.get('members', 0):,} files" + (f" from {span}" if span else "")
        evidence = [
            ev("Size", fmt_bytes(a.size_bytes)),
            ev("Last accessed", f"{days:,} days ago"),
            ev("Users with access", a.shared_with),
            ev("Owner", f"{a.owner}" + (f" ({a.department})" if a.department else "")),
            ev("Storage class", a.storage_class),
            ev("Business value (heuristic)", f"{value} — based on access recency and audience"),
        ]
        if contents:
            evidence.insert(2, ev("Contains", contents))
        signals.append(Signal(
            kind="stale_large", category="stale",
            title=f"{a.name}: large and untouched for {days // 365} year{'s' if days >= 730 else ''}",
            summary="Low-value, high-cost data. Archiving to cold storage keeps it recoverable at ~4% of the cost.",
            target_ids=[a.id],
            evidence=evidence,
            details={"days_since_use": days, "value": value},
            confidence=round(min(0.95, 0.6 + days / 3650), 3),
            action="ARCHIVE",
        ))
    # cold prefixes in object storage (e.g. old data-lake partitions): many medium objects, one decision
    prefixes: dict[str, list] = {}
    for a in ctx.active(include_outputs=False):
        if a.source != "s3-inventory" or a.id in ctx.claimed:
            continue
        days = ctx.days_since_use(a)
        if days >= s.stale_days:
            prefix = "/".join(a.path.split("/")[:3])
            prefixes.setdefault(prefix, []).append((a, days))
    for prefix, objs in sorted(prefixes.items(), key=lambda kv: -sum(o.size_bytes for o, _ in kv[1])):
        total = sum(o.size_bytes for o, _ in objs)
        if len(objs) < 20 or total < 50 * 1024**3:
            continue
        ids = [o.id for o, _ in objs]
        ctx.claimed.update(ids)
        newest = min(d for _, d in objs)
        signals.append(Signal(
            kind="stale_large", category="stale",
            title=f"Cold prefix {prefix}/ — {len(objs):,} objects untouched for {newest // 30}+ months",
            summary="Every object under this prefix has gone unread for over a year. Move the prefix to a cold storage tier.",
            target_ids=ids,
            evidence=[ev("Objects", f"{len(objs):,}"), ev("Total size", fmt_bytes(total)),
                      ev("Most recent access", f"{newest:,} days ago"),
                      ev("Oldest access", f"{max(d for _, d in objs):,} days ago"),
                      ev("Owners", ", ".join(sorted({o.owner for o, _ in objs}))[:120])],
            details={"prefix": prefix, "bulk": True},
            confidence=round(min(0.95, 0.6 + newest / 3650), 3), action="ARCHIVE",
        ))
    rest = [(a, d) for a, d in rest_pool if a.id not in ctx.claimed]
    if rest:
        ids = [a.id for a, _ in rest]
        ctx.claimed.update(ids)
        total = sum(a.size_bytes for a, _ in rest)
        signals.append(Signal(
            kind="stale_large", category="stale",
            title=f"{len(rest)} more large objects untouched for over a year",
            summary="Bulk-archive candidates: large, cold and rarely shared.",
            target_ids=ids,
            evidence=[ev("Objects", len(rest)), ev("Total size", fmt_bytes(total)),
                      ev("Oldest access", f"{max(d for _, d in rest):,} days ago")],
            details={"bulk": True},
            confidence=0.8, action="ARCHIVE",
        ))
    return signals
