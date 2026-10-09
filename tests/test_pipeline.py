import pandas as pd

from app.pipeline.executor import execute
from app.pipeline.planner import rules_plan
from app.pipeline.spec import PipelineSpec
from app.pipeline.validator import validate_spec
from app.scanner.profiler import DatasetView, profile_dataframe

DF = pd.DataFrame({
    "full_name": ["  Rahul Sharma ", "Rahul Sharma", "Priya Iyer", "Amit Shah", "Neha  Joshi"],
    "email": ["Rahul@Gmail.com", "rahul@gmail.com", "priya@x.in", None, "neha at x.in"],
    "phone": ["+91 98765 43210", "09876543210", "9123456789", "9000000000", "12345"],
    "signup_date": ["2024-01-05", "05/01/2024", "12 Mar 2024", "2024-02-30", "2024-03-01"],
    "pan": ["", "", "ABCPI1234F", "ABCPS1111A", "ABCPJ2222B"],
})
RULES = {"required_fields": ["email"], "formats": {"email": "email"},
         "dedupe": {"keys": ["email"], "keep": "most_complete", "never_merge_on_conflict": ["pan"]}}


def view():
    v = DatasetView("a", "t", DF.copy())
    profile_dataframe(v)
    return v


def spec(*steps):
    return {"steps": [{"id": f"s{i}", "op": op, "params": p} for i, (op, p) in enumerate(steps, 1)]}


def test_validator_rejects_unknown_ops_and_columns():
    v = view()
    rep = validate_spec(spec(("run_python", {"code": "import os"})), list(v.df.columns), v.profiles)
    assert not rep.valid and "unsupported operation" in rep.errors[0]
    rep = validate_spec(spec(("normalize_phone", {"columns": ["mobile_no"]})), list(v.df.columns), v.profiles)
    assert not rep.valid and "does not exist" in rep.errors[0]
    rep = validate_spec(spec(("normalize_phone", {"columns": ["PHONE"]})), list(v.df.columns), v.profiles)
    assert rep.valid and rep.spec.steps[0].params["columns"] == ["phone"]  # case-insensitive resolution


def test_end_to_end_cleaning_counts_are_real():
    v = view()
    rep = validate_spec(spec(
        ("trim_whitespace", {"columns": ["*"]}),
        ("normalize_email", {"columns": ["email"]}),
        ("normalize_phone", {"columns": ["phone"]}),
        ("parse_dates", {"columns": ["signup_date"]}),
        ("deduplicate", {"keys": ["email"], "keep": "most_complete", "conflict_columns": ["pan"]}),
        ("validate_rules", {"required": ["email"], "formats": {"email": "email", "phone": "phone"}}),
    ), list(v.df.columns), v.profiles)
    assert rep.valid, rep.errors
    res = execute(v.df, rep.spec)
    out = res.output
    assert res.totals["rows_in"] == 5
    assert res.totals["duplicates_removed"] == 1                       # the two Rahul rows merged
    assert set(out["phone"]) >= {"+919876543210", "+919123456789"}
    assert "2024-01-05" in set(out["signup_date"])                     # 05/01/2024 read day-first
    reasons = " ".join(res.quarantine["__reason"].tolist())
    assert "REQUIRED_MISSING: email" in reasons                       # Amit has no email
    assert "INVALID_FORMAT: email" in reasons                         # "neha at x.in"
    assert res.totals["rows_out"] + res.totals["rows_quarantined"] + res.totals["duplicates_removed"] == 5


def test_rules_planner_builds_a_valid_spec_from_plain_english():
    v = view()
    draft = rules_plan("Trim whitespace, normalize emails, standardize phone numbers to +91 format, "
                       "convert dates to YYYY-MM-DD, remove duplicates and quarantine rows missing email", v, RULES)
    ops = [s.op for s in draft.steps]
    assert ops == ["trim_whitespace", "normalize_email", "normalize_phone", "parse_dates", "deduplicate", "validate_rules"]
    dedupe = next(s for s in draft.steps if s.op == "deduplicate")
    assert dedupe.params["keys"] == ["email"] and dedupe.params["conflict_columns"] == ["pan"]
    rep = validate_spec(PipelineSpec(steps=draft.steps), list(v.df.columns), v.profiles)
    assert rep.valid, rep.errors


def test_rules_planner_asks_instead_of_guessing():
    v = view()
    draft = rules_plan("make it look nice and predict churn", v, {})
    assert draft.questions and not draft.steps


def test_repair_text_operator_and_garbled_prompt():
    df = pd.DataFrame({
        "product_name": ["Air Cooler??????(White)  ", "  Cooler (Black)", "cafÃ© mug??????(Red)"],
        "product_price": ["  1200", "1500  ", "800"],
        "Review": ["Great product??????Loved it!", "Nice!", "cafÃ© vibes"],
    })
    v = DatasetView("p", "prod", df)
    profile_dataframe(v)
    assert any(i.code == "garbled_text" for i in v.issues)

    # Prompt from the user's screenshot:
    draft = rules_plan("Remove the ??? character from the product name and fix the dataset", v, {})
    assert not draft.unsupported, f"Should map all instructions: {draft.unsupported}"
    ops = [s.op for s in draft.steps]
    assert "repair_text" in ops
    assert "trim_whitespace" in ops

    rep = validate_spec(PipelineSpec(steps=draft.steps), list(v.df.columns), v.profiles)
    assert rep.valid, rep.errors

    res = execute(v.df, rep.spec)
    out = res.output
    # Check that ??? is gone and mojibake is repaired
    assert "Air Cooler (White)" in out["product_name"].iloc[0]
    assert "???" not in out["product_name"].iloc[0]
    assert "café" in out["product_name"].iloc[2]
    # Check that whitespace was trimmed
    assert out["product_price"].iloc[0] == "1200"
    assert out["product_price"].iloc[1] == "1500"

