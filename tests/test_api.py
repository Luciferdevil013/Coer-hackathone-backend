"""End-to-end API tests on the generated demo estates."""
from pathlib import Path

from app.services.storage import Layout


def findings(client, ws, **params):
    r = client.get(f"/api/workspaces/{ws}/findings", params=params)
    assert r.status_code == 200
    return r.json()


def test_scan_produces_scored_findings(client, demo):
    s = client.get(f"/api/workspaces/{demo['acme']}/summary").json()
    assert s["has_scan"] and 0 < s["health_score"] < 80
    assert s["totals"]["total_bytes"] > 4 * 1024**4  # ~4.6 TB incl. S3 inventory
    kinds = {f["kind"] for f in findings(client, demo["acme"])}
    assert {"secret_exposed", "sensitive_data", "redundant_backup", "near_duplicate", "version_chain", "exact_duplicate",
            "stale_large", "unused_columns", "duplicate_rows", "data_quality"} <= kinds


def test_tenant_isolation(client, demo):
    acme_f = findings(client, demo["acme"])[0]
    r = client.get(f"/api/workspaces/{demo['zenith']}/findings/{acme_f['id']}")
    assert r.status_code == 404
    asset = client.get(f"/api/workspaces/{demo['acme']}/assets", params={"limit": 1}).json()[0]
    r = client.post(f"/api/workspaces/{demo['zenith']}/pipelines/plan", json={"asset_id": asset["id"], "prompt": "trim whitespace"})
    assert r.status_code == 404
    acme_rules = client.get(f"/api/workspaces/{demo['acme']}/rules").json()["active"]["rules"]
    zen_rules = client.get(f"/api/workspaces/{demo['zenith']}/rules").json()["active"]["rules"]
    assert acme_rules["dedupe"]["keys"] != zen_rules["dedupe"]["keys"]


def test_destructive_actions_need_confirmation(client, demo):
    f = next(x for x in findings(client, demo["acme"]) if x["kind"] == "redundant_backup" and "orders" in x["title"])
    r = client.post(f"/api/workspaces/{demo['acme']}/findings/{f['id']}/actions", json={"action": "APPLY"})
    assert r.status_code == 409


def test_apply_and_undo_backup_plan(client, demo):
    ws = demo["acme"]
    f = next(x for x in findings(client, ws) if x["kind"] == "redundant_backup" and "orders" in x["title"])
    lay = Layout(ws)
    before = sorted(p.name for p in (lay.files / "backups").iterdir())
    r = client.post(f"/api/workspaces/{ws}/findings/{f['id']}/actions", json={"action": "APPLY", "confirm": True})
    assert r.status_code == 200, r.text
    after = sorted(p.name for p in (lay.files / "backups").iterdir())
    assert after == ["orders_backup_04.csv", "orders_backup_05.csv"]
    assert (lay.trash / "backups/orders_backup_03.csv").exists()
    assert (lay.archive / "backups/orders_backup_01.csv").exists()
    r = client.post(f"/api/workspaces/{ws}/actions/{r.json()['action_id']}/undo")
    assert r.status_code == 200, r.text
    assert sorted(p.name for p in (lay.files / "backups").iterdir()) == before
    assert client.get(f"/api/workspaces/{ws}/findings/{f['id']}").json()["status"] == "open"


def test_secure_moves_to_owner_only_vault(client, demo):
    ws = demo["acme"]
    f = next(x for x in findings(client, ws) if x["kind"] == "secret_exposed")
    r = client.post(f"/api/workspaces/{ws}/findings/{f['id']}/actions", json={"action": "APPLY", "confirm": True})
    assert r.status_code == 200, r.text
    vault_file = Layout(ws).secured / "config/.env.bak"
    assert vault_file.exists() and oct(vault_file.stat().st_mode & 0o777) == "0o600"
    summary = client.get(f"/api/workspaces/{ws}/summary").json()
    assert summary["recommendations"]["secrets"] == 0


def test_explanation_falls_back_to_template(client, demo):
    f = findings(client, demo["acme"])[0]
    r = client.post(f"/api/workspaces/{demo['acme']}/findings/{f['id']}/explain").json()
    assert r["source"] == "template" and "Recommended action" in r["explanation"]


def test_clean_studio_plan_dry_run_commit(client, demo):
    ws = demo["acme"]
    asset = next(a for a in client.get(f"/api/workspaces/{ws}/assets", params={"q": "crm/customers_master"}).json())
    dup = next(x for x in findings(client, ws) if x["kind"] == "duplicate_rows" and asset["id"] in x["target_asset_ids"])
    detail = client.get(f"/api/workspaces/{ws}/findings/{dup['id']}").json()
    mergeable = [c["cluster_id"] for c in detail["details"]["clusters"] if not c["conflict_columns"] and not c["exact"]][:5]
    conflict = next(c["cluster_id"] for c in detail["details"]["clusters"] if c["conflict_columns"])
    r = client.post(f"/api/workspaces/{ws}/findings/{dup['id']}/row-decisions", json={"decisions": {conflict: "merge"}})
    assert r.status_code == 400  # conflicting PANs can never be merged
    r = client.post(f"/api/workspaces/{ws}/findings/{dup['id']}/row-decisions",
                    json={"decisions": {cid: "merge" for cid in mergeable}})
    assert r.status_code == 200 and r.json()["counts"]["merge"] == len(mergeable)

    plan = client.post(f"/api/workspaces/{ws}/pipelines/plan", json={
        "asset_id": asset["id"],
        "prompt": "Trim whitespace in all columns, normalize emails, standardize phone numbers to +91 format, "
                  "convert dates to YYYY-MM-DD, remove duplicate records by email and quarantine rows missing email"}).json()
    assert plan["validation"]["valid"], plan["validation"]
    ops = [s["op"] for s in plan["spec"]["steps"]]
    assert "merge_rows" in ops and "deduplicate" in ops and plan["planner"] == "rules"
    assert ops.index("merge_rows") < ops.index("deduplicate")  # reviewed merges take precedence

    run = client.post(f"/api/workspaces/{ws}/pipelines/dry-run", json={
        "asset_id": asset["id"], "spec": plan["spec"], "prompt": "test", "planner": plan["planner"]}).json()
    t = run["metrics"]["totals"]
    assert run["status"] == "dry_run" and t["rows_in"] == 7917
    assert t["rows_out"] + t["rows_quarantined"] + t["duplicates_removed"] == t["rows_in"]
    assert t["rows_quarantined"] >= 30  # conflicting-PAN groups + missing/invalid emails
    merge_step = next(s for s in run["metrics"]["steps"] if s["op"] == "merge_rows")
    assert merge_step["rows_removed"] >= len(mergeable)  # every approved group actually merged
    assert not Path(Layout(ws).outputs / "customers_master.cleaned.csv").exists()  # dry run never writes outputs

    done = client.post(f"/api/workspaces/{ws}/runs/{run['id']}/commit").json()
    assert done["status"] == "committed" and done["output_asset_id"]
    assert (Layout(ws).outputs / "customers_master.cleaned.csv").exists()
    open_clean = {x["kind"] for x in findings(client, ws) if x["action"] == "CLEAN" and asset["id"] in x["target_asset_ids"]}
    assert "duplicate_rows" not in open_clean and "data_quality" not in open_clean
    assert "anomalous_values" in open_clean  # this pipeline did not quarantine anomalies, so that finding stays open
    events = [e["event"] for e in client.get(f"/api/workspaces/{ws}/audit").json()]
    assert "pipeline.commit" in events and "pipeline.dry_run" in events


def test_security_view_and_bulk_apply(client, demo):
    ws = demo["acme"]
    sec = client.get(f"/api/workspaces/{ws}/security").json()
    labels = {e["label"] for e in sec["entities"]}
    assert {"Aadhaar number", "PAN"} <= labels and sec["rows"]
    preview = client.post(f"/api/workspaces/{ws}/findings/bulk-apply", json={"min_confidence": 0.95}).json()
    assert preview["would_apply"] >= 1
    done = client.post(f"/api/workspaces/{ws}/findings/bulk-apply", json={"min_confidence": 0.95, "confirm": True}).json()
    assert done["applied"] >= 1
    hist = client.get(f"/api/workspaces/{ws}/summary").json()["history"]
    assert len(hist) >= 2 and hist[-1]["score"] >= hist[0]["score"]
    listed = {f["id"]: f for f in findings(client, ws, status="resolved")}
    assert any(f["action_id"] for f in listed.values())  # inline undo handle


def test_anomaly_finding_and_quarantine_pipeline(client, demo):
    ws = demo["acme"]
    f = next(x for x in findings(client, ws) if x["kind"] == "anomalous_values" and "customers_master" in x["title"])
    detail = client.get(f"/api/workspaces/{ws}/findings/{f['id']}").json()
    flagged = {(fl["column"], fl["value"]) for r in detail["details"]["rows"] for fl in r["flags"]}
    assert {("age", 250.0), ("age", 999.0), ("lifetime_value_inr", 98500000.0)} <= flagged
    plan = client.post(f"/api/workspaces/{ws}/pipelines/plan", json={"asset_id": f["target_asset_ids"][0],
                       "prompt": detail["details"]["suggested_prompt"], "use_review": False}).json()
    assert [s["op"] for s in plan["spec"]["steps"]] == ["quarantine_anomalies"] and plan["validation"]["valid"]
    health = client.get("/api/health").json()["ml"]
    assert health["isolation_forest"]["ready"] and health["rapidfuzz"]["ready"]
