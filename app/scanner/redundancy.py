"""Redundancy detectors: backup series, version chains / near-duplicate datasets, exact duplicates.

Order matters: backups claim their assets first, then byte-identical duplicates, then the
similarity graph (version chains / near-duplicate datasets). An asset is recommended by at most one redundancy finding, so actions never
conflict.
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime

from app.engine.signals import ScanContext, Signal, ev
from app.models import Asset
from app.scanner.profiler import norm_col
from app.scanner.readers import effective_ext

# ----------------------------------------------------------------- name normalisation

_VERSION_WORDS = {
    "final", "finale", "fin", "latest", "new", "newer", "newest", "real", "copy", "updated", "update",
    "revised", "rev", "draft", "old", "fixed", "ok", "approved", "edited", "modified", "last", "current",
    "version", "ver", "v", "actual", "use", "this", "one", "corrected", "clean", "cleaned",
}
_SPLIT = re.compile(r"[^a-z0-9]+")
_ALNUM = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")


def _stem(name: str) -> str:
    ext = effective_ext(name)
    low = name.lower()
    idx = low.rfind("." + ext) if ext else -1
    return low[:idx] if idx > 0 else low


def version_base(name: str) -> tuple[str, int]:
    """'Report_FINAL_REAL_NEW.xlsx' -> ('report', 3). Returns (base, number of version tokens removed).

    Years and standalone numbers are kept ('sales_2023' vs 'sales_2024', 'campaign_07' vs
    'campaign_08' never collapse). Digits only count as a version when glued to a version word
    ('final2', 'v3') or directly following one ('final_2').
    """
    stem = _stem(name)
    removed = len(re.findall(r"\(\d+\)", stem))          # "file (1).csv" browser-download copies
    stem = re.sub(r"\(\d+\)", " ", stem)
    kept: list[str] = []
    prev_version = False
    for part in (p for p in _SPLIT.split(stem) if p):
        if part.isdigit():
            if prev_version and len(part) <= 2:
                removed += 1
            else:
                kept.append(part)
                prev_version = False
            continue
        subs = [t for t in _ALNUM.split(part) if t]
        has_word = any(t in _VERSION_WORDS for t in subs)
        for t in subs:
            if t in _VERSION_WORDS or re.fullmatch(r"v\d+", t) or (t.isdigit() and len(t) <= 2 and has_word):
                removed += 1
                prev_version = True
            else:
                kept.append(t)
                prev_version = False
    return "_".join(kept), removed


_BACKUP_RX = re.compile(r"(backup|bkp|bak|snapshot|dump)", re.I)


def backup_base(a: Asset) -> str | None:
    in_backup_dir = any(_BACKUP_RX.search(p) for p in a.path.lower().split("/")[:-1])
    stem = _stem(a.name)
    if not (_BACKUP_RX.search(stem) or in_backup_dir):
        return None
    base = re.sub(r"\d{4}[-_]?\d{2}[-_]?\d{2}", "", stem)
    base = re.sub(r"\d+", "", base)
    base = re.sub(r"[^a-z]+", "_", base).strip("_")
    folder = "/".join(a.path.lower().split("/")[:-1])
    return f"{a.source}:{folder}:{base}:{effective_ext(a.name)}"


# ----------------------------------------------------------------- similarity


_TYPE_FAMILY = {"integer": "num", "decimal": "num", "date": "date", "email": "email", "phone": "phone",
                "boolean": "bool", "text": "text", "mixed": "text", "empty": "empty"}


def match_columns(ctx: ScanContext, va, vb) -> list[tuple[str, str, str, float]]:
    """Map columns of two datasets: (col_a, col_b, method, score).

    1. same normalised name   2. same detected PII entity (email<->email, phone<->phone)
    3. all-MiniLM-L6-v2 similarity of the header names, only between type-compatible columns.
    """
    from app.ml.embeddings import get_semantic, humanize

    pairs: list[tuple[str, str, str, float]] = []
    left, right = list(va.df.columns), list(vb.df.columns)
    na, nb = {c: norm_col(c) for c in left}, {c: norm_col(c) for c in right}
    for a in left:
        for b in right:
            if na[a] == nb[b] and b not in {p[1] for p in pairs}:
                pairs.append((a, b, "name", 1.0))
                break
    used_a, used_b = {p[0] for p in pairs}, {p[1] for p in pairs}
    for a in left:
        ea = va.profiles[a].pii_entity if a in va.profiles else None
        if a in used_a or not ea:
            continue
        for b in right:
            if b not in used_b and b in vb.profiles and vb.profiles[b].pii_entity == ea:
                pairs.append((a, b, "pii-type", 1.0))
                used_a.add(a), used_b.add(b)
                break
    rest_a, rest_b = [a for a in left if a not in used_a], [b for b in right if b not in used_b]
    sem = get_semantic()
    if rest_a and rest_b and sem.available:
        m = sem.matrix([humanize(a) for a in rest_a], [humanize(b) for b in rest_b])
        cands = sorted(((float(m[i, j]), i, j) for i in range(len(rest_a)) for j in range(len(rest_b))), reverse=True)
        for score, i, j in cands:
            if score < ctx.settings.semantic_column_threshold:
                break
            a, b = rest_a[i], rest_b[j]
            if a in used_a or b in used_b:
                continue
            # MiniLM header scores are modest: require a mutual best match that clearly beats the runner-up
            row_2nd = max([m[i, k] for k in range(len(rest_b)) if k != j], default=0.0)
            col_2nd = max([m[k, j] for k in range(len(rest_a)) if k != i], default=0.0)
            if m[i].argmax() != j or m[:, j].argmax() != i or score - max(row_2nd, col_2nd) < 0.08:
                continue
            ta = _TYPE_FAMILY.get(va.profiles[a].inferred_type, "text")
            tb = _TYPE_FAMILY.get(vb.profiles[b].inferred_type, "text")
            if ta != tb:
                continue
            pairs.append((a, b, "semantic", round(score, 3)))
            used_a.add(a), used_b.add(b)
    return pairs


def tabular_similarity(ctx: ScanContext, a_id: str, b_id: str) -> dict | None:
    va, vb = ctx.views.get(a_id), ctx.views.get(b_id)
    if va is None or vb is None:
        return None
    key = tuple(sorted((a_id, b_id)))
    if key in ctx.sim_cache:
        return ctx.sim_cache[key]
    pairs = match_columns(ctx, va, vb)
    n_union = len(va.df.columns) + len(vb.df.columns) - len(pairs)
    schema_j = len(pairs) / n_union if n_union else 0.0
    out = {"score": round(schema_j * 0.3, 4), "schema": round(schema_j, 4), "rows": 0.0, "containment": 0.0}
    if schema_j >= 0.5 and pairs:
        pairs.sort()
        ha = va.row_hashes(tuple(p[0] for p in pairs))
        hb = vb.row_hashes(tuple(p[1] for p in pairs))
        if ha and hb:
            inter = len(ha & hb)
            row_j = inter / len(ha | hb)
            out = {"score": round(0.3 * schema_j + 0.7 * row_j, 4), "schema": round(schema_j, 4),
                   "rows": round(row_j, 4), "containment": round(inter / min(len(ha), len(hb)), 4)}
    sem = [{"a": a, "b": b, "score": s} for a, b, m, s in pairs if m == "semantic"]
    if sem:
        out["semantic_columns"] = sem
    ctx.sim_cache[key] = out
    return out


def _shingles(text: str, k: int = 5) -> set[int]:
    words = re.findall(r"\w+", text.lower())
    return {hash(" ".join(words[i:i + k])) for i in range(max(0, len(words) - k + 1))}


def doc_vector(ctx: ScanContext, asset_id: str):
    from app.ml.embeddings import get_semantic

    if asset_id not in ctx.doc_vectors:
        sem = get_semantic()
        text = ctx.texts.get(asset_id) or ""
        ctx.doc_vectors[asset_id] = sem.document_vector(text) if sem.available and text else None
    return ctx.doc_vectors[asset_id]


def text_similarity(ctx: ScanContext, a_id: str, b_id: str) -> dict | None:
    """Exact wording (5-word shingles) plus meaning (MiniLM document embeddings).
    Reworded copies have low shingle overlap but very high semantic similarity."""
    ta, tb = ctx.texts.get(a_id), ctx.texts.get(b_id)
    if not ta or not tb:
        return None
    sa, sb = _shingles(ta), _shingles(tb)
    if not sa or not sb:
        return None
    j = len(sa & sb) / len(sa | sb)
    out = {"score": round(j, 4), "containment": round(len(sa & sb) / min(len(sa), len(sb)), 4)}
    vx, vy = doc_vector(ctx, a_id), doc_vector(ctx, b_id)
    if vx is not None and vy is not None:
        cos = float(vx @ vy)
        out["semantic"] = round(cos, 4)
        if cos > j:  # meaning matches better than wording -> paraphrased copy (never "identical": containment stays low)
            out["score"] = round(cos, 4)
            out["method"] = "semantic"
    return out


def similarity(ctx: ScanContext, a: Asset, b: Asset) -> dict | None:
    if a.content_hash and a.content_hash == b.content_hash:
        return {"score": 1.0, "identical": True, "containment": 1.0}
    if a.kind == "tabular" and b.kind == "tabular":
        return tabular_similarity(ctx, a.id, b.id)
    if a.id in ctx.texts and b.id in ctx.texts:
        return text_similarity(ctx, a.id, b.id)
    return None


def _ts(a: Asset) -> datetime:
    return a.modified_at or datetime.min


def _fmt_date(d: datetime | None) -> str:
    return d.strftime("%d %b %Y") if d else "unknown"


# ----------------------------------------------------------------- backups


def detect_backups(ctx: ScanContext) -> list[Signal]:
    groups: dict[str, list[Asset]] = defaultdict(list)
    for a in ctx.active(include_outputs=False):
        if a.id in ctx.claimed:
            continue
        key = backup_base(a)
        if key:
            groups[key].append(a)

    signals = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        members.sort(key=_ts, reverse=True)
        latest = members[0]
        plan = [{"asset_id": latest.id, "action": "KEEP", "reason": "Latest restore point"}]
        distinct_kept = [latest]
        rows_ev = []
        for m in members[1:]:
            match = None
            for k in distinct_kept:
                sim = similarity(ctx, m, k)
                if sim and (sim.get("identical") or (sim["score"] >= 0.999 and sim.get("containment", 0) >= 0.999)):
                    match = (k, sim)
                    break
            if match:
                pct = "100%" if match[1].get("identical") else f"{match[1]['score'] * 100:.2f}%"
                plan.append({"asset_id": m.id, "action": "DELETE",
                             "reason": f"{pct} identical to {match[0].name}"})
                rows_ev.append(f"{m.name} → {pct} identical")
            elif len(distinct_kept) < 2:
                distinct_kept.append(m)
                plan.append({"asset_id": m.id, "action": "KEEP", "reason": "Previous distinct restore point (retention: 2)"})
                rows_ev.append(f"{m.name} → useful (previous distinct version)")
            else:
                plan.append({"asset_id": m.id, "action": "ARCHIVE",
                             "reason": f"Obsolete: {len(distinct_kept)} newer distinct backups exist"})
                rows_ev.append(f"{m.name} → obsolete")
        targets = [p["asset_id"] for p in plan if p["action"] != "KEEP"]
        if not targets:
            continue
        for m in members:
            ctx.claimed.add(m.id)
        n_del = sum(1 for p in plan if p["action"] == "DELETE")
        n_arc = sum(1 for p in plan if p["action"] == "ARCHIVE")
        stem = members[0].name
        signals.append(Signal(
            kind="redundant_backup", category="redundant",
            title=f"{len(members)} backups in the '{stem.split('.')[0].rstrip('0123456789_- ')}' series, {len(targets)} redundant",
            summary=f"{n_del} identical copies can be deleted and {n_arc} obsolete backups archived; the latest 2 distinct restore points are kept.",
            target_ids=targets,
            keep_ids=[p["asset_id"] for p in plan if p["action"] == "KEEP"],
            plan=plan,
            evidence=[
                ev("Backups in series", len(members)),
                ev("Latest backup", f"{latest.name} ({_fmt_date(latest.modified_at)})"),
                ev("Identical copies", n_del),
                ev("Obsolete (beyond retention of 2 distinct)", n_arc),
                *[ev("Comparison", r) for r in rows_ev[:6]],
            ],
            details={"series": key, "member_ids": [m.id for m in members]},
            confidence=0.97 if n_del and not n_arc else 0.9,
            action="DELETE" if n_del >= n_arc else "ARCHIVE",
        ))
    return signals


# ----------------------------------------------------------------- version chains & near duplicates


class _UF:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def detect_similar_clusters(ctx: ScanContext) -> list[Signal]:
    s = ctx.settings
    candidates = [a for a in ctx.active(include_outputs=False) if a.id not in ctx.claimed and a.kind in ("tabular", "document", "text", "archive", "media", "binary")]
    by_base: dict[tuple, list[Asset]] = defaultdict(list)
    for a in candidates:
        base, removed = version_base(a.name)
        if base:  # versions live next to each other: same connector, folder, base name and kind
            folder = a.path.rsplit("/", 1)[0] if "/" in a.path else ""
            by_base[(a.source, folder, base, a.kind)].append(a)

    uf = _UF()
    edges: dict[tuple[str, str], dict] = {}
    name_linked: set[tuple[str, str]] = set()

    # (1) same version-base name, confirmed by content when content exists
    for (_, _folder, base, kind), members in by_base.items():
        if len(members) < 2 or not any(version_base(m.name)[1] for m in members):
            continue  # at least one file must carry a version marker
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                sim = similarity(ctx, a, b)
                if sim is None:
                    if not (a.content_available or b.content_available):
                        sim = {"score": 0.6, "name_only": True}  # metadata-only: lower confidence
                    else:
                        continue
                if sim.get("name_only") or sim["score"] >= 0.3:
                    key = tuple(sorted((a.id, b.id)))
                    edges[key] = sim
                    name_linked.add(key)
                    uf.union(a.id, b.id)

    # (2) content near-duplicates regardless of name (tabular only - cheap and precise)
    tab = [a for a in candidates if a.kind == "tabular" and a.id in ctx.views]
    for i, a in enumerate(tab):
        for b in tab[i + 1:]:
            key = tuple(sorted((a.id, b.id)))
            if key in edges:
                continue
            sim = tabular_similarity(ctx, a.id, b.id)
            if sim and sim["score"] >= s.near_dup_threshold:
                edges[key] = sim
                uf.union(a.id, b.id)

    # (3) reworded documents: all-MiniLM-L6-v2 meaning-level similarity, regardless of name
    docs = [a for a in candidates if a.kind in ("document", "text") and a.id in ctx.texts and doc_vector(ctx, a.id) is not None]
    for i, a in enumerate(docs):
        for b in docs[i + 1:]:
            key = tuple(sorted((a.id, b.id)))
            if key in edges:
                continue
            la, lb = len(ctx.texts[a.id].split()), len(ctx.texts[b.id].split())
            if min(la, lb) < 0.5 * max(la, lb):
                continue  # a summary is not a copy: require comparable length
            sim = text_similarity(ctx, a.id, b.id)
            if sim and sim.get("semantic", 0) >= s.semantic_doc_threshold:
                edges[key] = sim
                uf.union(a.id, b.id)

    clusters: dict[str, list[Asset]] = defaultdict(list)
    for a in candidates:
        if a.id in uf.p:
            clusters[uf.find(a.id)].append(a)

    signals = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        ids = {m.id for m in members}
        all_name_linked = all(
            tuple(sorted((x, y))) in name_linked or tuple(sorted((x, y))) not in edges
            for x in ids for y in ids if x < y
        ) and len({version_base(m.name)[0] for m in members}) == 1
        members.sort(key=lambda m: (_ts(m), version_base(m.name)[1]))
        keep = members[-1]
        chain = []
        plan = [{"asset_id": keep.id, "action": "KEEP", "reason": "Latest version"}]
        sims = []
        for m in members:
            sim = edges.get(tuple(sorted((m.id, keep.id)))) or similarity(ctx, m, keep)
            if m.id != keep.id and sim and not sim.get("name_only"):
                sims.append(sim["score"])
            chain.append({
                "asset_id": m.id, "name": m.name, "modified": _fmt_date(m.modified_at), "size_bytes": m.size_bytes,
                "similarity_to_latest": None if m.id == keep.id or not sim or sim.get("name_only") else sim["score"],
            })
            if m.id == keep.id:
                continue
            if sim and (sim.get("identical") or (sim["score"] >= 0.995 and sim.get("containment", 0) >= 0.999)):
                plan.append({"asset_id": m.id, "action": "DELETE", "reason": "Identical content to the kept version"})
            else:
                plan.append({"asset_id": m.id, "action": "ARCHIVE", "reason": "Superseded by a newer version"})
        targets = [p["asset_id"] for p in plan if p["action"] != "KEEP"]
        ctx.claimed.update(ids)
        n_del = sum(1 for p in plan if p["action"] == "DELETE")
        n_arc = len(targets) - n_del
        avg_sim = sum(sims) / len(sims) if sims else None
        name_only = not sims
        kind = "version_chain" if all_name_linked else "near_duplicate"
        base = version_base(keep.name)[0]
        if kind == "version_chain":
            title = f"Version chain of {len(members)} files: '{base}'"
            summary = f"Latest meaningful version is {keep.name}. {n_arc} older versions can be archived" + (
                f" and {n_del} identical copies deleted." if n_del else ".")
        else:
            noun = "datasets" if all(m.kind == "tabular" for m in members) else "documents"
            title = f"{len(members)} near-duplicate {noun} ({avg_sim * 100:.1f}% similar)" if avg_sim else f"{len(members)} near-duplicate {noun}"
            summary = f"These look like versions of the same {noun[:-1]}. Keep {keep.name}, archive {n_arc}" + (
                f", delete {n_del} duplicate." if n_del else ".")
        conf = 0.6 if name_only else min(0.98, 0.55 + 0.45 * (avg_sim or 0))
        member_ids = {m.id for m in members}
        cluster_edges = [e for k, e in edges.items() if k[0] in member_ids and k[1] in member_ids]
        sem_cols = {f"{c['a']} ↔ {c['b']}" for e in cluster_edges for c in e.get("semantic_columns", [])}
        sem_docs = [e["semantic"] for e in cluster_edges if e.get("method") == "semantic"]
        method = "file-name version pattern" if name_only else "file names + content comparison"
        if sem_docs:
            method = "meaning-level text similarity (all-MiniLM-L6-v2) + wording overlap"
        elif sem_cols:
            method = "row-level content + semantic header matching (all-MiniLM-L6-v2)"
        signals.append(Signal(
            kind=kind, category="redundant", title=title, summary=summary,
            target_ids=targets, keep_ids=[keep.id], plan=plan,
            evidence=[
                ev("Files in group", len(members)),
                ev("Kept (latest)", f"{keep.name} ({_fmt_date(keep.modified_at)})"),
                ev("Average similarity to latest", f"{avg_sim * 100:.1f}%" if avg_sim is not None else "name pattern only (no content)"),
                ev("Method", method),
                ev("Identical copies", n_del),
                *([ev("Headers matched by meaning", ", ".join(sorted(sem_cols)))] if sem_cols else []),
                *([ev("Semantic similarity (MiniLM)", f"{max(sem_docs) * 100:.1f}%")] if sem_docs else []),
            ],
            details={"chain": chain, "base_name": base, "semantic_columns": sorted(sem_cols)},
            confidence=round(conf, 3),
            action="ARCHIVE" if n_arc >= n_del else "DELETE",
        ))
    return signals


# ----------------------------------------------------------------- exact duplicates


_CANON_PENALTY = re.compile(r"(copy|backup|old|tmp|temp|duplicate|\(\d+\)|downloads?)", re.I)


def detect_exact_duplicates(ctx: ScanContext) -> list[Signal]:
    """Byte-identical copies. Assets already handled by a backup/version finding can act as the
    kept copy: those plans always keep at least one copy of every distinct content, so deleting
    the unclaimed duplicates never loses data."""
    groups: dict[str, list[Asset]] = defaultdict(list)
    for a in ctx.active(include_outputs=False):
        if a.content_hash and a.size_bytes > 0:
            groups[a.content_hash].append(a)
    signals = []
    for h, members in groups.items():
        free = [m for m in members if m.id not in ctx.claimed]
        if len(members) < 2 or not free:
            continue
        anchored = [m for m in members if m.id in ctx.claimed]
        free.sort(key=lambda m: (len(_CANON_PENALTY.findall(m.path)), m.path.count("/"),
                                 -_ts(m).timestamp() if m.modified_at else 0))
        if anchored:
            keep, dups = anchored[0], free
            keep_reason = "Identical copy already managed by another recommendation"
        else:
            keep, dups = free[0], free[1:]
            keep_reason = "Canonical location"
        if not dups:
            continue
        ctx.claimed.update(m.id for m in dups)  # the kept copy stays free so it can anchor a version chain
        signals.append(Signal(
            kind="exact_duplicate", category="redundant",
            title=f"{len(dups) + 1} identical copies of {keep.name}",
            summary=f"Byte-for-byte identical (same content hash). Keep {keep.path}, delete {len(dups)} cop{'y' if len(dups) == 1 else 'ies'}.",
            target_ids=[d.id for d in dups], keep_ids=[keep.id],
            plan=[{"asset_id": keep.id, "action": "KEEP", "reason": keep_reason}]
                 + [{"asset_id": d.id, "action": "DELETE", "reason": "Identical content hash"} for d in dups],
            evidence=[
                ev("Identical copies", len(dups) + 1),
                ev("Content hash (BLAKE2b)" if keep.source != "s3-inventory" else "ETag", h[:16] + "…"),
                ev("Kept", keep.path),
                ev("Duplicates", ", ".join(d.path for d in dups[:5])),
            ],
            details={"hash": h},
            confidence=0.99,
            action="DELETE",
        ))
    return signals
