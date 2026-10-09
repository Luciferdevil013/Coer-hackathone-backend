"""Free-text datasets (tweets, reviews, tickets): duplicates, hidden PII, HTML codes, empty columns, .xls."""
import pandas as pd

from app.pipeline import executor
from app.pipeline.planner import rules_plan, suggest_prompt
from app.pipeline.spec import PipelineSpec
from app.scanner import pii, readers, textnorm
from app.scanner.profiler import DatasetView, profile_dataframe
from app.scanner.rows import find_text_duplicate_clusters

TWEETS = [
    "Need #recipes? Visit the Kraft page &amp; watch the video http://t.co/abc123",
    "Need #recipes? Visit the Kraft page &amp; watch the video http://t.co/abc123",       # exact copy
    "need #recipes?  visit the kraft page & watch the video http://t.co/abc123",          # case/space/HTML
    "RT @kraft: Need #recipes? Visit the Kraft page &amp; watch the video http://t.co/zz9",  # retweet, other link
    "Please email me at mama4life07@gmail.com about a review for my blog",
    "Talk to us, we delighted Kraft too! Call 01484 682687 today",
    "Coupon http://pinterest.com/pin/446911962998255057/ for cheese lovers everywhere",
    "lol",
    "lol",
] + [f"Completely different tweet number {i} about the super bowl game" for i in range(60)]


def _view() -> DatasetView:
    df = pd.DataFrame({"ID": [str(i) for i in range(len(TWEETS))], "text": TWEETS,
                       "Unnamed: 2": ["@ChiefKeef"] + [None] * (len(TWEETS) - 1)})
    v = DatasetView("a1", "tweets.xlsx", df)
    profile_dataframe(v)
    return v


def test_text_normalisation_levels():
    a, b = TWEETS[0], TWEETS[3]
    assert textnorm.basic(TWEETS[0]) == textnorm.basic(TWEETS[2].replace("  ", " "))
    assert textnorm.content(a) == textnorm.content(b) != ""
    assert textnorm.content("lol") == ""  # too short to call a copy


def test_free_text_profile_html_and_empty_column():
    v = _view()
    p = v.profiles["text"]
    assert p.free_text and not v.profiles["ID"].free_text
    assert p.html_entities == 3
    codes = {i.code: i for i in v.issues}
    assert "html_entities" in codes and "empty_column" in codes
    assert codes["empty_column"].column == "Unnamed: 2"
    prompt = suggest_prompt(v).lower()
    assert "decode html codes in 'text'" in prompt
    # 'Unnamed: 2' still holds a value (text that spilled over from the tweet), so it is never dropped unasked
    assert "unnamed: 2" not in prompt
    q = rules_plan("drop empty columns", v, {})
    assert not any(s.op == "drop_columns" for s in q.steps) and any("Unnamed: 2" in x for x in q.questions)


def test_pii_inside_free_text_without_url_false_positives():
    v = _view()
    p = v.profiles["text"]
    assert p.text_pii == {"EMAIL": 1, "PHONE": 1}  # the pinterest pin id is not a card
    assert p.text_pii_rows == 2
    assert "mama4life07" not in " ".join(p.samples)
    assert not pii.scan_text("Order 0123456789 shipped").entities
    assert pii.scan_text("contact :rpandey@malupaper.com91-9767897282").entities == {"EMAIL": 1, "PHONE": 1}


def test_text_duplicates_graded_by_level():
    v = _view()
    clusters, keys = find_text_duplicate_clusters(v)
    assert keys["text"] == "text"
    st = v.text_duplicates
    assert (st["exact"], st["basic"], st["content"], st["retweets"]) == (1, 1, 1, 1)
    assert len(clusters) == 1 and clusters[0]["row_ids"] == [0, 1, 2, 3] and not clusters[0]["exact"]


def test_suggested_clean_runs_end_to_end():
    v = _view()
    find_text_duplicate_clusters(v)
    draft = rules_plan(suggest_prompt(v) + " and redact personal data in 'text' and drop the 'Unnamed: 2' column", v, {})
    ops = [s.op for s in draft.steps]
    assert {"decode_html", "deduplicate", "drop_columns", "redact_text_pii"} <= set(ops), ops
    assert "mask_pii" not in ops  # prose keeps its text; only the values inside are masked
    res = executor.execute(v.df, PipelineSpec(source_asset_id="a1", steps=draft.steps))
    out = res.output
    assert len(out) == len(TWEETS) - 3 and "Unnamed: 2" not in out.columns
    joined = " ".join(out["text"])
    assert "&amp;" not in joined and "mama4life07@gmail.com" not in joined and "01484 682687" not in joined
    assert "lol" in joined  # short texts are never merged


def test_xls_is_tabular_and_failures_are_explained(tmp_path):
    assert readers.detect_kind("Unstructured Data Japanese.xls") == "tabular"
    bad = tmp_path / "broken.xls"
    bad.write_bytes(b"not really a workbook")
    errs: list[str] = []
    assert readers.read_table(bad, 100, errs) is None
    assert errs and errs[0].startswith("Could not read the XLS file")


def test_keyless_rows_compare_everything_but_a_real_id():
    # unique business values (amounts, account numbers) make rows different; only a row counter is ignored
    n = 60
    df = pd.DataFrame({"Sno": [str(i + 1) for i in range(n)], "vendor": ["Acme Supplies"] * n,
                       "account_number": [str(100000000 + i * 7) for i in range(n)],
                       "amount_inr": [str(1000 + i) for i in range(n)]})
    df.loc[n - 1, ["account_number", "amount_inr"]] = df.loc[0, ["account_number", "amount_inr"]].tolist()
    v = DatasetView("a2", "vendor_payments.csv", df)
    profile_dataframe(v)
    clusters, _ = find_text_duplicate_clusters(v)
    assert v.text_duplicates["mode"] == "row" and "Sno" not in v.text_duplicates["columns"]
    assert [c["row_ids"] for c in clusters] == [[0, n - 1]]
