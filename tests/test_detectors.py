import pandas as pd

from app.scanner import pii
from app.scanner.profiler import DatasetView, profile_dataframe
from app.scanner.redundancy import version_base
from app.scanner.rows import find_duplicate_clusters, score_pair


def test_verhoeff_roundtrip_and_aadhaar():
    base = "23456789012"
    num = base + pii.verhoeff_check_digit(base)
    assert pii.verhoeff_valid(num)
    assert pii.is_aadhaar(f"{num[:4]} {num[4:8]} {num[8:]}")
    tampered = num[:-1] + str((int(num[-1]) + 1) % 10)
    assert not pii.is_aadhaar(tampered)
    assert not pii.is_aadhaar("1" + num[1:])  # cannot start with 0/1


def test_pan_card_phone_validators():
    assert pii.is_pan("ABCPE1234F")
    assert not pii.is_pan("ABCDE1234F")  # 4th char must be a valid holder type
    assert pii.is_card("4111 1111 1111 1111")
    assert not pii.is_card("4111 1111 1111 1112")
    for ok in ("+91 98765 43210", "09876543210", "9876543210", "+91-9876543210"):
        assert pii.is_phone_in(ok), ok
    for bad in ("12345", "5876543210", "98765-4321x"):
        assert not pii.is_phone_in(bad), bad


def test_column_classification_and_masking():
    base = "34567890123"
    vals = [base + pii.verhoeff_check_digit(base)] * 10
    c = pii.classify_column("aadhaar_no", vals)
    assert c and c.entity == "AADHAAR"
    assert pii.mask_value("AADHAAR", vals[0]).endswith(vals[0][-4:]) and "X" in pii.mask_value("AADHAAR", vals[0])
    assert pii.classify_column("city", ["Mumbai", "Delhi"]) is None


def test_secret_scan_and_redaction():
    text = "AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP\nDB_PASSWORD=hunter2hunter"
    scan = pii.scan_text(text)
    assert scan.secrets["AWS access key"] == 1
    assert "AKIAABCDEFGHIJKLMNOP" not in pii.mask_text(text)


def test_version_base():
    assert version_base("Report_FINAL_REAL_NEW.xlsx")[0] == "report"
    assert version_base("Report_final2.xlsx")[0] == "report"
    assert version_base("customers (1).csv") == ("customers", 1)
    # years and standalone numbers are not versions
    assert version_base("sales_2023.csv")[0] != version_base("sales_2024.csv")[0]
    assert version_base("campaign_07_master.mp4")[0] != version_base("campaign_08_master.mp4")[0]


def test_name_alone_is_never_a_duplicate():
    a = {"email": "", "phone": "", "name": "rahul sharma", "verified": {}}
    b = {"email": "", "phone": "", "name": "rahul sharma", "verified": {}}
    assert score_pair(a, b)[0] == 0.0


def test_duplicate_clusters_with_conflict():
    df = pd.DataFrame({
        "full_name": ["Rahul", "Rahul Sharma", "Priya Iyer", "Priya Iyer", "Amit Shah"],
        "email": ["rahul@gmail.com", "RAHUL@gmail.com ", "priya@x.in", "priya@x.in", "amit@x.in"],
        "phone": ["9876543210", "+91 98765 43210", "9123456789", "9123456789", "9000000000"],
        "pan": ["", "", "ABCPI1234F", "XYZPI9999Q", "ABCPS1111A"],
    })
    v = DatasetView("a1", "t", df)
    profile_dataframe(v)
    clusters, keys = find_duplicate_clusters(v, 0.8)
    assert keys["email"] == "email" and keys["phone"] == "phone"
    by_rows = {tuple(c["row_ids"]): c for c in clusters}
    assert (0, 1) in by_rows and not by_rows[(0, 1)]["conflict_columns"]
    assert (2, 3) in by_rows and by_rows[(2, 3)]["conflict_columns"] == ["pan"]
    assert all(4 not in c["row_ids"] for c in clusters)
