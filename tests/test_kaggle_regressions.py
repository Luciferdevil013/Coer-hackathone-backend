"""Regressions found by running Clean Studio on the Kaggle test estate (test-data/run_clean_suite.py)."""
import time

import pandas as pd

from app.pipeline import operators as O
from app.pipeline import spec as S
from app.pipeline import virtual as V
from app.pipeline.executor import execute
from app.pipeline.planner import rules_plan, suggest_prompt
from app.pipeline.spec import PipelineSpec
from app.scanner.profiler import DatasetView, profile_dataframe
from app.services.workspaces import DEFAULT_RULES


def view(df: pd.DataFrame, label: str = "t.csv") -> DatasetView:
    v = DatasetView("a", label, df)
    profile_dataframe(v)
    return v


TEMPLATE = "I'm having an issue with the {product_purchased}. Please assist. Thanks for your help with this problem."
TICKETS = pd.DataFrame({
    "Ticket ID": [str(i) for i in range(1, 81)],
    "Customer Name": [f"Person {i}" for i in range(80)],
    "Customer Email": [f"user{i % 30}@example.com" for i in range(80)],   # synthetic emails repeat across people
    "Product Purchased": [["Xbox", "LG TV", "Dell XPS"][i % 3] for i in range(80)],
    "Ticket Description": [TEMPLATE] * 80,                                   # templated text repeats too
})


def test_event_table_is_never_merged_by_email_or_template_text():
    v = view(TICKETS, "customer_support_tickets.csv")
    assert "duplicate records by" not in suggest_prompt(v).lower()
    d = rules_plan("remove duplicate rows", v, DEFAULT_RULES)
    assert any("Not deduplicating by Customer Email" in a for a in d.assumptions)
    res = execute(TICKETS, PipelineSpec(steps=d.steps))
    assert res.totals["duplicates_removed"] == 0  # 80 different tickets stay 80 tickets
    asked = rules_plan("remove duplicates by Customer Email", v, DEFAULT_RULES)
    assert any(q.startswith("Check before running") for q in asked.questions)  # explicit request: allowed, but flagged


def test_all_columns_never_fuzzy_matches_one_column():
    df = pd.DataFrame({"full_text": ["  hello there  ", "x "] * 30, "trailing_whitespace": ["[true, false]"] * 60})
    d = rules_plan("Trim whitespace in all columns", view(df), {})
    assert [s.params["columns"] for s in d.steps if s.op == "trim_whitespace"] == [["*"]]


def test_spilled_over_text_column_is_not_dropped_unasked():
    df = pd.DataFrame({"v1": ["ham"] * 100, "v2": [f"message {i}" for i in range(100)],
                       "Unnamed: 2": [" but it continues here"] + [None] * 99})
    v = view(df)
    assert "unnamed" not in suggest_prompt(v).lower()


def test_impact_warns_when_rows_with_their_own_ticket_id_are_merged():
    spec = PipelineSpec(steps=[{"id": "s1", "op": "deduplicate", "params": {"keys": ["Customer Email"]}}])
    res = execute(TICKETS, spec)
    tree = V.build("r", "t.csv", TICKETS, res, {})
    imp = V.impact(tree, [s.model_dump() for s in spec.steps])
    chk = next(c for c in imp["checks"] if c["name"].startswith("Unique IDs merged away"))
    assert chk["status"] == "warn" and "Ticket ID" in chk["name"] and "50 rows" in chk["detail"]


def test_dropping_a_column_that_holds_values_is_called_out():
    df = pd.DataFrame({"a": [str(i) for i in range(10)], "b": ["x"] + [None] * 9})
    spec = PipelineSpec(steps=[{"id": "s1", "op": "drop_columns", "params": {"columns": ["b"]}}])
    tree = V.build("r", "t.csv", df, execute(df, spec), {})
    lost = next(c for c in V.impact(tree, [])["checks"] if c["name"] == "Values lost with dropped columns")
    assert "'b' still held 1 value" in lost["detail"]


def test_large_dedupe_is_fast():
    n = 120_000  # Flipkart (205k rows, 34k merges) took 103 s with per-row .loc writes
    # rows i and i+80k share a key; each half of a pair is missing the value the other one has
    df = pd.DataFrame({O.ROW_ID: range(n), "k": [str(i % 80_000) for i in range(n)],
                       "a": [None if i >= 80_000 else "x" for i in range(n)],
                       "b": [None if i < 40_000 else "y" for i in range(n)]})
    t = time.time()
    r = O.deduplicate(df, S.DedupeParams(keys=["k"]), None)
    assert r.rows_removed == 40_000 and time.time() - t < 20
    assert r.df["a"].notna().all() and r.df["b"].notna().all()  # survivors were coalesced from their duplicates


def test_review_dataset_never_suggests_deduping_identical_reviews():
    df = pd.DataFrame({
        "product_name": ["Boat Earphones", "Boat Earphones", "Boat Earphones"],
        "product_price": [999, 999, 999],
        "Rate": [5, 5, 5],
        "Review": ["good product", "good product", "great bass"],
        "Summary": ["good", "good", "great"],
        "Sentiment": ["positive", "positive", "positive"],
    })
    v = view(df, "Dataset-SA.csv")
    dup_issue = next((i for i in v.issues if i.code == "duplicate_rows"), None)
    assert dup_issue is not None
    assert dup_issue.suggested_op is None
    assert "common for review" in dup_issue.message
    assert "remove duplicate rows" not in suggest_prompt(v).lower()
