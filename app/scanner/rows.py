"""Row-level duplicate detection (exact + fuzzy) and dataset quality findings.

Matching follows the DataPilot design (§7.2): normalise, block on trusted keys (email/phone),
score with explicit weights, never treat a name alone as identity, and flag conflicting
verified IDs instead of silently merging.
"""
from __future__ import annotations

import re
from collections import defaultdict

from rapidfuzz import fuzz

from app.engine.signals import ScanContext, Signal, ev
from app.scanner import pii, textnorm
from app.scanner.profiler import DatasetView, is_event_dataset, norm_col

WEIGHTS = {"email": 0.45, "phone": 0.35, "name": 0.20}
MAX_BLOCK = 50
MAX_CLUSTERS_SHOWN = 200
MAX_ROWS_SHOWN = 20      # per group in the review UI; row_ids always carry the full group


def _norm_email(v) -> str:
    return str(v).strip().lower() if v is not None and str(v).strip() else ""


def _norm_phone(v) -> str:
    d = re.sub(r"\D", "", str(v)) if v is not None else ""
    return d[-10:] if len(d) >= 10 else ""


def _norm_name(v) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(v).lower())).strip() if v is not None else ""


def key_columns(view: DatasetView) -> dict[str, str | list[str] | None]:
    email = phone = name = None
    verified: list[str] = []
    for col, p in view.profiles.items():
        nc = col.lower()
        if p.pii_entity == "EMAIL" and email is None:
            email = col
        elif p.pii_entity == "PHONE" and phone is None:
            phone = col
        elif p.pii_entity in ("PAN", "AADHAAR"):
            verified.append(col)
        if name is None and re.search(r"(^|_)(full_?)?name($|_)", re.sub(r"[^a-z0-9]+", "_", nc)) and p.inferred_type in ("text", "mixed"):
            name = col
    return {"email": email, "phone": phone, "name": name, "verified": verified}


class _UF:
    def __init__(self):
        self.p: dict[int, int] = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def score_pair(r1: dict, r2: dict) -> tuple[float, list[str]]:
    num = den = 0.0
    reasons = []
    for k in ("email", "phone"):
        if r1[k] and r2[k]:
            den += WEIGHTS[k]
            if r1[k] == r2[k]:
                num += WEIGHTS[k]
                reasons.append(f"same {k}")
    if r1["name"] and r2["name"]:
        den += WEIGHTS["name"]
        sim = fuzz.token_set_ratio(r1["name"], r2["name"]) / 100
        num += WEIGHTS["name"] * sim
        reasons.append(f"name similarity {sim * 100:.0f}%")
    if den < 0.5:  # not enough trusted evidence (e.g. only a name) -> never a match
        return 0.0, reasons
    return num / den, reasons


def find_duplicate_clusters(view: DatasetView, threshold: float) -> tuple[list[dict], dict]:
    keys = key_columns(view)
    if not (keys["email"] or keys["phone"]):
        return [], keys
    df = view.df
    n = len(df)

    def col(c, fn):
        return [fn(v) for v in df[c].tolist()] if c else [""] * n

    emails, phones, names = col(keys["email"], _norm_email), col(keys["phone"], _norm_phone), col(keys["name"], _norm_name)
    verified_cols = {c: df[c].tolist() for c in keys["verified"]}
    recs = [
        {
            "email": emails[i], "phone": phones[i], "name": names[i],
            "verified": {c: str(vals[i]).strip().upper() for c, vals in verified_cols.items() if vals[i] not in (None, "")},
        }
        for i in range(n)
    ]
    blocks: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(recs):
        if r["email"]:
            blocks["e:" + r["email"]].append(i)
        if r["phone"]:
            blocks["p:" + r["phone"]].append(i)
    trimmed = df.fillna("").astype(str).apply(lambda s: s.str.strip())
    full_hash = trimmed.apply(lambda r: hash(tuple(r)), axis=1).tolist()

    uf = _UF()
    pair_scores: dict[tuple[int, int], tuple[float, list[str]]] = {}
    for members in blocks.values():
        if len(members) < 2 or len(members) > MAX_BLOCK:
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                i, j = members[x], members[y]
                if (i, j) in pair_scores:
                    continue
                if full_hash[i] == full_hash[j]:
                    sc, why = 1.0, ["identical row"]
                else:
                    sc, why = score_pair(recs[i], recs[j])
                pair_scores[(i, j)] = (sc, why)
                if sc >= threshold:
                    uf.union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in uf.p:
        groups[uf.find(i)].append(i)
    clusters = []
    for root, rows in groups.items():
        if len(rows) < 2:
            continue
        rows.sort()
        scores = [pair_scores[(a, b)] for a in rows for b in rows if a < b and (a, b) in pair_scores and pair_scores[(a, b)][0] >= threshold]
        conf = min(s for s, _ in scores) if scores else threshold
        reasons = sorted({w for _, ws in scores for w in ws})
        conflict_cols = []
        for c in keys["verified"]:
            vals = {recs[i]["verified"].get(c) for i in rows} - {None}
            if len(vals) > 1:
                conflict_cols.append(c)
        exact = bool(scores) and all(ws == ["identical row"] for _, ws in scores)
        clusters.append({
            "cluster_id": f"c{root}",
            "row_ids": rows,
            "confidence": round(conf, 3),
            "exact": exact,
            "reasons": reasons,
            "conflict_columns": conflict_cols,
            "decision": None,
        })
    # what needs a human first: conflicts, then fuzzy matches (largest groups first), then exact copies
    clusters.sort(key=lambda c: (not c["conflict_columns"], c["exact"], -len(c["row_ids"]), -c["confidence"]))
    return clusters, keys


# ---------------------------------------------------------------- duplicates without contact keys
# Files like tweets, reviews or tickets have no email/phone to match on. When the long free text
# *is* the record (tweets, transcripts, ticket bodies), copies are grouped on the normalised text
# (see textnorm). Otherwise a row is a duplicate only when every column apart from a surrogate ID
# matches - two reviewers both writing "Great product" are not duplicates.

LONG_TEXT_CHARS = 60
CHOICE_COLUMN_MAX = 20   # at most this many distinct answers: a multiple-choice column
TEXT_LEVELS = (
    (1.0, "identical text"),
    (0.97, "same text ignoring case, spacing and HTML codes"),
    (0.9, "same text once links, retweet prefixes (RT @user:) and punctuation are removed"),
)
ROW_LEVELS = (
    (1.0, "identical apart from the ID"),
    (0.97, "identical ignoring case, spacing and HTML codes"),
)


def text_key_column(view: DatasetView) -> str | None:
    cands = [c for c, p in view.profiles.items() if p.free_text and p.avg_length >= LONG_TEXT_CHARS]
    return max(cands, key=lambda c: view.profiles[c].avg_length * view.profiles[c].non_null) if cands else None


_ID_NAME = re.compile(r"^(unnamed_?\d*|s_?no|sr_?no|serial(_?no)?|index|idx|row(_?(id|no|num))?|uuid|guid)$|(^|_)(id|key)$")


def _is_surrogate_id(view: DatasetView, col: str) -> bool:
    """A row identifier that says nothing about the record: 'ID', 'Sno', 'Unnamed: 0', or a plain 1..n counter.
    Unique business values (PAN, account number, amount) are *not* IDs - they are what makes rows differ."""
    p = view.profiles[col]
    if p.pii_entity or p.free_text or p.non_null != view.rows or p.unique_ratio < 1.0:
        return False
    if _ID_NAME.search(norm_col(col)):
        return True
    if p.inferred_type == "integer":
        nums = sorted(int(x) for x in view.df[col].tolist())
        return nums[-1] - nums[0] == len(nums) - 1  # consecutive row counter
    return False


def duplicate_key_columns(view: DatasetView, allow_text: bool = True) -> tuple[list[str], str]:
    """(columns, mode) for keyless dedupe: ([long text column], "text") or (content columns, "row").
    allow_text=False for event tables (tickets, orders): there a templated description repeats across
    different events, so only whole rows can be duplicates."""
    from app.scanner.profiler import is_event_dataset

    if is_event_dataset(view):
        allow_text = False
    col = text_key_column(view) if allow_text else None
    if col:
        return [col], "text"
    from app.scanner.profiler import NEAR_EMPTY_RATE

    cols = [c for c, p in view.profiles.items()
            if not _is_surrogate_id(view, c) and p.null_rate < NEAR_EMPTY_RATE]
    # a survey or form where every column is a pick-list: two people giving the same answers are not duplicates
    if cols and all(view.profiles[c].distinct <= CHOICE_COLUMN_MAX for c in cols):
        return [], "row"
    return cols, "row"


def find_text_duplicate_clusters(view: DatasetView) -> tuple[list[dict], dict]:
    cols, mode = duplicate_key_columns(view)
    if not cols:
        return [], {}
    df = view.df[cols]
    if mode == "text":
        raw = df[cols[0]].tolist()
        exact_keys = [str(v).strip() if v is not None else "" for v in raw]
        basic_keys = [textnorm.basic(v) for v in raw]
        group_keys = [textnorm.content(v) for v in raw]
    else:
        recs = df.itertuples(index=False, name=None)
        exact_keys, basic_keys = [], []
        for r in recs:
            exact_keys.append(tuple("" if v is None else str(v).strip() for v in r))
            basic_keys.append(tuple(textnorm.basic(v) for v in r))
        group_keys = [k if any(k) else "" for k in basic_keys]
    levels = TEXT_LEVELS if mode == "text" else ROW_LEVELS
    groups: dict = defaultdict(list)
    for i, k in enumerate(group_keys):
        if k:
            groups[k].append(i)
    clusters, stats = [], {"exact": 0, "basic": 0, "content": 0, "retweets": 0}
    for rows_ in groups.values():
        if len(rows_) < 2:
            continue
        exact = {exact_keys[i] for i in rows_}
        basic = {basic_keys[i] for i in rows_}
        level = 0 if len(exact) == 1 else 1 if len(basic) == 1 else 2
        # rows beyond the first copy of each variant, per level
        stats["exact"] += len(rows_) - len(exact)
        stats["basic"] += len(exact) - len(basic)
        stats["content"] += len(basic) - 1
        clusters.append({
            "cluster_id": f"t{rows_[0]}", "row_ids": rows_, "confidence": levels[level][0],
            "exact": level == 0, "reasons": [levels[level][1]], "conflict_columns": [], "decision": None,
        })
    if mode == "text":
        stats["retweets"] = sum(1 for k in basic_keys if re.match(r"(rt|mt)\b", k))
    clusters.sort(key=lambda c: (c["exact"], -len(c["row_ids"])))
    view.text_duplicates = {"mode": mode, "columns": cols, "column": cols[0] if mode == "text" else None,
                            **stats, "groups": len(clusters)}
    return clusters, {"email": None, "phone": None, "name": None, "verified": [], "text": cols[0] if mode == "text" else None}


def _display_rows(view: DatasetView, row_ids: list[int], keys: dict) -> list[dict]:
    cols = [c for c in [keys.get("text"), keys.get("name"), keys.get("email"), keys.get("phone")] if c]
    cols += [c for c in view.df.columns if c not in cols][:6 - len(cols)]
    out = []
    for i in row_ids[:MAX_ROWS_SHOWN]:
        rec = {"_row": int(i)}
        for c in cols:
            v = view.df.iat[i, view.df.columns.get_loc(c)]
            p = view.profiles[c]
            ent = p.pii_entity
            if v is None or (isinstance(v, float) and v != v):  # None / NaN
                rec[c] = ""
            elif ent in ("AADHAAR", "PAN", "CREDIT_CARD", "BANK_ACCOUNT", "PASSWORD"):
                rec[c] = pii.mask_value(ent, str(v))
            elif p.free_text:
                rec[c] = pii.mask_text(str(v))[:280]
            else:
                rec[c] = str(v)
        out.append(rec)
    return out


def _quote(cols: list[str]) -> str:
    return ", ".join(f"'{c}'" for c in cols)


def dedupe_prompt(st: dict) -> str:
    if st["mode"] == "text":
        return f"Remove duplicate rows by '{st['column']}' ignoring links and retweet prefixes"
    return f"Remove duplicate rows by {_quote(st['columns'])}"


def _text_duplicate_signal(ctx: ScanContext, a, view: DatasetView) -> Signal | None:
    clusters, keys = find_text_duplicate_clusters(view)
    if not clusters:
        return None
    st = view.text_duplicates
    dup_rows = st["exact"] + st["basic"] + st["content"]
    # keep both review tabs populated: near-copies need a human most, but show exact groups too
    exact_c = [c for c in clusters if c["exact"]][:MAX_CLUSTERS_SHOWN // 4]
    shown = [c for c in clusters if not c["exact"]][:MAX_CLUSTERS_SHOWN - len(exact_c)] + exact_c
    for c in shown:
        c["rows"] = _display_rows(view, c["row_ids"], keys)
    if st["mode"] == "text":
        col = st["column"]
        parts = [f"{st['exact']:,} rows repeat another row's '{col}' exactly"]
        if st["basic"]:
            parts.append(f"{st['basic']:,} differ only in case, spacing or HTML codes")
        if st["content"]:
            parts.append(f"{st['content']:,} are the same text once links and retweet prefixes are removed")
        summary = "; ".join(parts) + (f". {st['retweets']:,} rows are retweets (RT @user:)." if st["retweets"] else ".")
        evidence = [
            ev("Rows scanned", f"{view.rows:,}"),
            ev("Compared on", f"'{col}' (free text; no email or phone column to match on)"),
            ev("Exact copies", f"{st['exact']:,} rows"),
            ev("Differ only in case / spacing / HTML codes", f"{st['basic']:,} rows"),
            ev("Near-copies (links, RT @user: and punctuation ignored)", f"{st['content']:,} rows"),
            ev("Duplicate groups", f"{len(clusters):,}"),
            ev("Retweets", f"{st['retweets']:,}"),
        ]
        title = f"{dup_rows:,} duplicate or near-copy rows in {a.name}"
    else:
        skipped = [c for c in view.df.columns if c not in st["columns"]]
        summary = (f"{st['exact']:,} rows are exact copies of another row"
                   + (f" apart from {_quote(skipped)}" if skipped else "")
                   + (f"; {st['basic']:,} more differ only in case, spacing or HTML codes" if st["basic"] else "") + ".")
        evidence = [
            ev("Rows scanned", f"{view.rows:,}"),
            ev("Compared on", f"all {len(st['columns'])} columns" + (f" except {_quote(skipped)} (ID or empty)" if skipped else "")),
            ev("Exact copies", f"{st['exact']:,} rows"),
            ev("Differ only in case / spacing / HTML codes", f"{st['basic']:,} rows"),
            ev("Duplicate groups", f"{len(clusters):,}"),
        ]
        title = f"{dup_rows:,} duplicate rows in {a.name}"

    is_event = is_event_dataset(view)
    if is_event:
        action = "KEEP"
        title = f"{dup_rows:,} identical review entries in {a.name} (Legitimate Feedback)"
        summary = (
            f"{dup_rows:,} rows share identical ratings and text across products. "
            "In review and event datasets, multiple users submitting common feedback "
            "(such as repeat 5-star ratings or short reviews like 'good product') "
            "represent genuine separate customer opinions. Removing them would distort review volume and rating distributions."
        )
        s_prompt = "Trim whitespace in all columns"
    else:
        action = "CLEAN"
        summary = summary + " Removing them keeps one copy of each."
        s_prompt = dedupe_prompt(st)

    return Signal(
        kind="duplicate_rows", category="quality", title=title,
        summary=summary,
        target_ids=[a.id], evidence=evidence,
        details={"clusters": shown, "total_clusters": len(clusters), "keys": keys, "dataset_rows": view.rows,
                 "text_stats": st, "match": st["mode"], "suggested_prompt": s_prompt},
        confidence=1.0 if is_event else round(sum(c["confidence"] * (len(c["row_ids"]) - 1) for c in clusters) / dup_rows, 3),
        action=action, records=dup_rows,
    )


def detect_duplicate_rows(ctx: ScanContext) -> list[Signal]:
    signals = []
    for asset_id, view in ctx.views.items():
        a = ctx.assets[asset_id]
        if a.status != "active" or a.source == "pipeline-output" or asset_id in ctx.doomed:
            continue
        clusters, keys = find_duplicate_clusters(view, ctx.settings.row_dup_threshold)
        if not (keys["email"] or keys["phone"]):
            sig = _text_duplicate_signal(ctx, a, view)
            if sig:
                signals.append(sig)
            continue
        if not clusters:
            continue
        dup_rows = sum(len(c["row_ids"]) - 1 for c in clusters)
        exact = sum(1 for c in clusters if c["exact"])
        fuzzy = len(clusters) - exact
        conflicts = sum(1 for c in clusters if c["conflict_columns"])
        shown = clusters[:MAX_CLUSTERS_SHOWN]
        for c in shown:
            c["rows"] = _display_rows(view, c["row_ids"], keys)
        avg_conf = sum(c["confidence"] for c in clusters) / len(clusters)
        signals.append(Signal(
            kind="duplicate_rows", category="quality",
            title=f"{dup_rows:,} duplicate records in {a.name}",
            summary=(f"{len(clusters):,} duplicate groups ({exact:,} exact, {fuzzy:,} likely). "
                     + (f"{conflicts} groups have conflicting verified IDs and will not be merged automatically. " if conflicts else "")
                     + "Review likely matches, then clean."),
            target_ids=[a.id],
            evidence=[
                ev("Rows scanned", f"{view.rows:,}"),
                ev("Duplicate groups", f"{len(clusters):,}"),
                ev("Exact duplicates", f"{exact:,} groups"),
                ev("Likely duplicates (fuzzy)", f"{fuzzy:,} groups"),
                ev("Match keys", ", ".join(k for k in (keys['email'], keys['phone']) if k) + (f" + name similarity ({keys['name']})" if keys["name"] else "")),
                ev("Conflicting verified IDs", conflicts),
                ev("Threshold", f"{ctx.settings.row_dup_threshold:.0%} weighted match score"),
            ],
            details={"clusters": shown, "total_clusters": len(clusters), "keys": keys,
                     "dataset_rows": view.rows, "weights": WEIGHTS},
            confidence=round(avg_conf, 3), action="CLEAN", records=dup_rows,
        ))
    return signals


def detect_quality(ctx: ScanContext) -> list[Signal]:
    from app.pipeline.planner import suggest_prompt

    signals = []
    for asset_id, view in ctx.views.items():
        a = ctx.assets[asset_id]
        if a.status != "active" or a.source == "pipeline-output" or asset_id in ctx.doomed or not view.issues or view.rows == 0:
            continue
        issues = [i for i in view.issues if i.code != "duplicate_rows"]
        if not issues:
            continue
        cells = view.rows * max(1, len(view.df.columns))
        affected = sum(i.count for i in issues)
        rate = affected / cells
        signals.append(Signal(
            kind="data_quality", category="quality",
            title=f"{affected:,} data-quality issues in {a.name}",
            summary="; ".join(i.message for i in issues[:3]) + ("…" if len(issues) > 3 else ""),
            target_ids=[a.id],
            evidence=[ev("Rows", f"{view.rows:,}"), ev("Columns", len(view.df.columns)),
                      ev("Cells affected", f"{affected:,} ({rate:.1%})")]
                     + [ev(i.code.replace("_", " ").title(), i.message) for i in issues[:6]],
            details={"issues": [i.__dict__ for i in issues], "issue_rate": round(rate, 4),
                     "suggested_prompt": suggest_prompt(view)},
            confidence=0.95, action="CLEAN", records=affected,
        ))
    return signals
