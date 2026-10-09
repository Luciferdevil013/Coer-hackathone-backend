import os
import sys
import tempfile
from pathlib import Path

import pytest

_TMP = tempfile.mkdtemp(prefix="dg-test-")
os.environ["DG_DATA_DIR"] = _TMP
os.environ.pop("ANTHROPIC_API_KEY", None)  # tests exercise the deterministic paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def demo(client):
    r = client.post("/api/demo/setup")
    assert r.status_code == 200, r.text
    wss = {w["name"]: w["id"] for w in r.json()["workspaces"]}
    return {"acme": wss["Acme Retail Pvt Ltd"], "zenith": wss["Zenith Finance Ltd"]}
