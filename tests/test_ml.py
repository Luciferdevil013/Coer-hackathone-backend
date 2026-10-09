import numpy as np
import pandas as pd
import pytest

from app.ml import anomaly
from app.ml.embeddings import get_semantic
from app.pipeline.executor import execute
from app.pipeline.spec import PipelineSpec, Step
from app.pipeline.validator import resolve_column
from app.scanner.profiler import DatasetView, profile_dataframe


def _frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "phone": [f"98{rng.integers(10**7, 10**8)}" for _ in range(n)],
        "age": rng.integers(19, 72, n).astype(str),
        "lifetime_value_inr": rng.lognormal(9.6, 0.5, n).round().astype(int).astype(str),
    })
    df.loc[10, "age"] = "250"
    df.loc[20, "lifetime_value_inr"] = "98500000"
    return df


def test_numeric_columns_skip_ids_and_phones():
    df = _frame()
    v = DatasetView("x", "t", df)
    profile_dataframe(v)
    assert anomaly.numeric_columns(df, v.profiles) == ["age", "lifetime_value_inr"]


def test_isolation_forest_flags_planted_values_only():
    df = _frame()
    res = anomaly.detect(df, ["age", "lifetime_value_inr"])
    flagged = {r["row"]: {f["column"] for f in r["flags"]} for r in res.rows}
    assert flagged.get(10) == {"age"} and flagged.get(20) == {"lifetime_value_inr"}
    assert len(flagged) <= 6  # precise: forest AND robust z must agree
    again = anomaly.detect(df, ["age", "lifetime_value_inr"])
    assert [r["row"] for r in again.rows] == [r["row"] for r in res.rows]  # deterministic (random_state)


def test_quarantine_anomalies_operator():
    spec = PipelineSpec(steps=[Step(id="s1", op="quarantine_anomalies", params={"columns": ["*"], "z_threshold": 3.5})])
    res = execute(_frame(), spec)
    reasons = " ".join(res.quarantine["__reason"])
    assert "ANOMALY: age=250" in reasons and "ANOMALY: lifetime_value_inr=9.85e+07" in reasons
    assert res.totals["rows_out"] + res.totals["rows_quarantined"] == 600


def test_rapidfuzz_column_resolution():
    cols = ["full_name", "phone", "signup_date"]
    assert resolve_column("phone_number", cols) == "phone"
    assert resolve_column("signupdate", cols) == "signup_date"
    assert resolve_column("salary", cols) is None


sem = get_semantic()
needs_model = pytest.mark.skipif(not sem.available, reason="all-MiniLM-L6-v2 not downloaded (python -m scripts.download_models)")


@needs_model
def test_semantic_similarity_orders_meaning():
    assert sem.similarity("contact number", "phone") > sem.similarity("contact number", "city")
    assert sem.similarity("town", "city") > sem.similarity("town", "email")


@needs_model
def test_semantic_header_matching_links_renamed_exports():
    from datetime import datetime

    from app.config import get_settings
    from app.engine.signals import ScanContext
    from app.scanner.redundancy import match_columns

    a = pd.DataFrame({"name": ["Rahul Sharma"] * 3, "email": ["r@x.in"] * 3, "phone": ["+919876543210"] * 3, "city": ["Pune"] * 3})
    b = a.rename(columns={"name": "Customer Name", "email": "E-mail Address", "phone": "Contact Number", "city": "Town"})
    va, vb = DatasetView("a", "a", a), DatasetView("b", "b", b)
    profile_dataframe(va), profile_dataframe(vb)
    ctx = ScanContext(workspace_id="w", settings=get_settings(), now=datetime.now(), assets={})
    pairs = {(x, y) for x, y, _, _ in match_columns(ctx, va, vb)}
    assert {("name", "Customer Name"), ("email", "E-mail Address"), ("phone", "Contact Number"), ("city", "Town")} <= pairs


@needs_model
def test_paraphrased_documents_are_semantically_close():
    a = "Customer personal data must be deleted within thirty days after the account is closed. " * 4
    b = "Personal information about customers shall be erased no later than 30 days once their account has been closed. " * 4
    c = "The vendor shall deliver goods as per the purchase order and agreed service levels. " * 4
    va, vb, vc = (sem.document_vector(t) for t in (a, b, c))
    assert float(va @ vb) > 0.85 > float(va @ vc)
