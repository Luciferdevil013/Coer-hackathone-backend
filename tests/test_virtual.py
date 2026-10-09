"""Virtual Data Tree: reconciliation into a Data-Patch, and the vault-backed in-place commit + rollback."""
from pathlib import Path

import pandas as pd

from app.pipeline import virtual as V
from app.pipeline.executor import execute
from app.pipeline.spec import PipelineSpec
from app.services.storage import Layout

DF = pd.DataFrame({
    "name": ["  Rahul Sharma ", "Rahul Sharma", "Priya Iyer", "Amit Shah", "Neha Joshi"],
    "email": ["Rahul@Gmail.com", "rahul@gmail.com", "priya@x.in", "amit@x.in", "neha@x.in"],
    "pan": ["ABCPS1234F", "ABCPS1234F", "ABCPI1234F", "ABCPS1111A", "ABCPJ2222B"],
    "aadhaar": ["234567890123", "234567890999", "345678901234", "", "456789012345"],
})


def run(*steps):
    spec = PipelineSpec(steps=[{"id": f"s{i}", "op": op, "params": p} for i, (op, p) in enumerate(steps, 1)])
    res = execute(DF, spec)
    return V.build("r1", "t.csv", DF, res, {"aadhaar": "AADHAAR", "pan": "PAN"}), res


def test_patch_has_one_exact_entry_per_change():
    tree, res = run(("trim_whitespace", {"columns": ["name"]}), ("normalize_email", {"columns": ["email"]}),
                    ("deduplicate", {"keys": ["email"], "keep": "most_complete"}),
                    ("mask_pii", {"columns": ["aadhaar"], "method": "partial"}))
    ops = [(p["row_id"], p["op"], p.get("col")) for p in tree.patch]
    assert (1, "DELETE", None) in ops                       # row 1 merged into row 0
    assert next(p for p in tree.patch if p["op"] == "DELETE")["kept_row"] == 0
    assert (0, "NORMALIZE", "name") in ops and (0, "NORMALIZE", "email") in ops
    assert {r for r, op, c in ops if op == "REDACT"} == {0, 2, 4}  # empty aadhaar (row 3) is not a change
    assert not any(r == 1 and op != "DELETE" for r, op, _ in ops)  # a deleted row only carries its DELETE
    m = V.mutations(tree, 100, 90)
    assert (m["pruned_duplicates"], m["masked_pii_cells"], m["storage_delta_bytes"]) == (1, 3, -10)
    assert len(res.output) + len(res.quarantine) + m["pruned_duplicates"] == len(DF)


def test_redaction_wins_over_earlier_normalization_and_display_never_leaks():
    tree, _ = run(("trim_whitespace", {"columns": ["*"]}), ("mask_pii", {"columns": ["aadhaar"], "method": "partial"}))
    red = next(p for p in tree.patch if p["op"] == "REDACT" and p["row_id"] == 0)
    assert red["operator"] == "mask_pii"
    shown = V.inspector(tree, "REDACT", 0, 10)
    cell = shown["rows"][0]["cells"]["aadhaar"]
    assert "234567890123" not in str(shown) and cell["before"] != cell["after"] and "•" in cell["before"]
    assert shown["rows"][0]["values"]["pan"].startswith("XXXXX")  # unchanged sensitive cells are masked too


def test_conflicting_duplicates_are_quarantined_not_deleted_and_impact_counts_them():
    df_steps = [("deduplicate", {"keys": ["pan"], "keep": "most_complete", "conflict_columns": ["aadhaar"]})]
    tree, res = run(*df_steps)
    assert [p["op"] for p in tree.patch] == ["QUARANTINE", "QUARANTINE"]  # same PAN, different Aadhaar: never merged
    imp = V.impact(tree, [{"op": op, "params": p} for op, p in df_steps])
    acct = next(c for c in imp["checks"] if c["name"] == "Row accounting")
    ent = next(c for c in imp["checks"] if c["name"].startswith("Entity preservation"))
    assert acct["status"] == "pass" and ent["status"] == "pass" and imp["safe"]


def test_shield_keeps_shape_but_hides_identifier():
    assert V.shield("+91 98765 43210") == "+•• ••••• •••••"
    assert V.shield_text("call me at 9876543210 today") == "call me at •••••••••• today"


# ---------------------------------------------------------------- API: dry run -> inspect -> commit in place -> rollback


def test_in_place_commit_is_atomic_vaulted_and_rolls_back(client, demo):
    ws = demo["zenith"]
    asset = next(a for a in client.get(f"/api/workspaces/{ws}/assets", params={"q": "loan_applicants"}).json())
    path = Layout(ws).files / asset["path"]
    original = path.read_bytes()
    spec = {"steps": [
        {"id": "s1", "op": "normalize_phone", "params": {"columns": ["phone"]}},
        {"id": "s2", "op": "deduplicate", "params": {"keys": ["pan"], "keep": "most_complete", "conflict_columns": ["aadhaar"]}},
        {"id": "s3", "op": "mask_pii", "params": {"columns": ["aadhaar"], "method": "partial"}},
    ]}
    r = client.post(f"/api/workspaces/{ws}/pipelines/dry-run", json={"asset_id": asset["id"], "spec": spec}).json()
    vdt = r["metrics"]["vdt"]
    assert vdt["in_place"]["supported"] and vdt["patch_entries"] > 0
    m = vdt["virtual_mutations"]
    assert m["masked_pii_cells"] > 0 and m["normalized_cells"] > 0
    assert path.read_bytes() == original  # the dry run never touches the real file

    diff = client.get(f"/api/workspaces/{ws}/runs/{r['id']}/diff", params={"op": "REDACT", "limit": 5}).json()
    assert diff["counts"]["REDACT"] == m["masked_pii_cells"] and len(diff["rows"]) == 5
    raw = pd.read_csv(path, dtype=str)["aadhaar"].dropna().tolist()[:50]
    assert not any(a in str(diff) for a in raw)  # no raw Aadhaar leaves the server
    imp = client.get(f"/api/workspaces/{ws}/runs/{r['id']}/impact").json()
    assert imp["safe"] and any(c["name"] == "PII exposure (aadhaar)" and c["status"] == "pass" for c in imp["checks"])
    patch = client.get(f"/api/workspaces/{ws}/runs/{r['id']}/patch")
    assert patch.status_code == 200 and len(patch.json()["patches"]) == vdt["patch_entries"]

    done = client.post(f"/api/workspaces/{ws}/runs/{r['id']}/commit", json={"mode": "in_place"}).json()
    vc = done["vault_commit"]
    assert done["status"] == "committed" and vc["codec"] in ("zstd", "gzip") and vc["vault_bytes"] < vc["original_bytes"]
    assert path.read_bytes() != original and not list(path.parent.glob(".*.tmp"))
    after = pd.read_csv(path, dtype=str)
    assert len(after) == r["metrics"]["totals"]["rows_out"]
    assert after["aadhaar"].dropna().str.startswith("X").all()
    assert (Layout(ws).vault / Path(vc["vault_file"]).name).exists()
    assert client.get(f"/api/workspaces/{ws}/commits/{vc['id']}/verify").json()["ok"]

    # a second dry run against the committed file, then the file changes underneath it: commit must refuse
    r2 = client.post(f"/api/workspaces/{ws}/pipelines/dry-run", json={"asset_id": asset["id"], "spec": {"steps": [
        {"id": "s1", "op": "trim_whitespace", "params": {"columns": ["*"]}}]}}).json()
    committed_bytes = path.read_bytes()
    path.write_bytes(committed_bytes + b"\n")
    stale = client.post(f"/api/workspaces/{ws}/runs/{r2['id']}/commit", json={"mode": "in_place"})
    assert stale.status_code == 400 and "stale" in stale.json()["detail"]
    blocked = client.post(f"/api/workspaces/{ws}/commits/{vc['id']}/rollback")
    assert blocked.status_code == 400  # the file no longer matches the committed version
    path.write_bytes(committed_bytes)
    assert client.post(f"/api/workspaces/{ws}/runs/{r2['id']}/discard").json()["status"] == "discarded"

    rb = client.post(f"/api/workspaces/{ws}/commits/{vc['id']}/rollback").json()
    assert rb["status"] == "rolled_back" and path.read_bytes() == original  # byte-for-byte restore
    assert client.post(f"/api/workspaces/{ws}/commits/{vc['id']}/rollback").status_code == 400
    events = [e["event"] for e in client.get(f"/api/workspaces/{ws}/audit").json()]
    assert {"vault.commit", "vault.rollback", "pipeline.discard"} <= set(events)
    journal = Layout(ws).journal.read_text().splitlines()
    assert [l for l in journal if vc["id"] in l][-1].find('"rollback"') > 0
