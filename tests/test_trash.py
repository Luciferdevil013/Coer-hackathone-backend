"""Deleting files by hand: always to the restorable trash, one undoable action per file."""
from app.services.storage import Layout


def test_trash_and_restore_files_by_hand(client, demo):
    ws = demo["acme"]
    W = f"/api/workspaces/{ws}"
    a, b = (next(x for x in client.get(f"{W}/assets", params={"q": q}).json()) for q in ("hr/employee_records", "backups/orders_backup_05"))
    files, trash = Layout(ws).files, Layout(ws).trash
    assert (files / a["path"]).exists()

    r = client.post(f"{W}/assets/trash", json={"asset_ids": [a["id"], b["id"]]}).json()
    assert r["trashed"] == 2 and len(r["action_ids"]) == 2
    assert not (files / a["path"]).exists() and (trash / a["path"]).exists()
    trashed = {x["id"] for x in client.get(f"{W}/assets", params={"status": "trashed"}).json()}
    assert {a["id"], b["id"]} <= trashed
    assert client.post(f"{W}/assets/trash", json={"asset_ids": [a["id"]]}).status_code == 400  # already in trash

    assert client.post(f"{W}/assets/{a['id']}/restore").status_code == 200  # each file restores on its own
    assert (files / a["path"]).exists() and not (trash / a["path"]).exists()
    assert (trash / b["path"]).exists()
    assert client.post(f"{W}/actions/{r['action_ids'][1]}/undo").status_code == 200  # or via Audit & Undo
    assert (files / b["path"]).exists()
    assert client.post(f"{W}/assets/{a['id']}/restore").status_code == 400  # not in trash any more
    events = [e["event"] for e in client.get(f"{W}/audit").json()]
    assert events.count("asset.trashed") >= 2 and "action.undo" in events
