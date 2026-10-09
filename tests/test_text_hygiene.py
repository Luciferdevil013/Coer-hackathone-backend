"""Column meaning (names, health, special categories), profile links, US phones and text-hygiene issues."""
import pandas as pd

from app.scanner import pii
from app.scanner.profiler import DatasetView, profile_dataframe


def issues(df: pd.DataFrame) -> dict[str, int]:
    v = DatasetView("a", "t", df)
    profile_dataframe(v)
    return {i.code: i.count for i in v.issues}, v


def test_name_columns_need_a_name_header():
    assert pii.classify_column("Customer Name", ["Asha Rao", "Vikram Singh"]).entity == "PERSON_NAME"
    assert pii.classify_column("Name", ["Bobby JacksOn", "Leslie Terry"]).entity == "PERSON_NAME"
    # values that look like names are not enough on their own
    assert pii.classify_column("city", ["Mumbai", "Delhi"]) is None
    assert pii.classify_column("product_name", ["Air Cooler", "Phone Case"]) is None
    assert pii.classify_column("Doctor", ["Matthew Smith"]) is None


def test_health_and_special_category_columns():
    assert pii.classify_column("Medical Condition", ["Cancer", "Asthma"]).entity == "HEALTH"
    assert pii.classify_column("blood_group", ["O+", "AB-"]).entity == "HEALTH"
    assert pii.classify_column("RaceDesc", ["White", "Asian"]).entity == "SPECIAL_CATEGORY"
    assert pii.classify_column("religion", ["Hindu", "Sikh"]).entity == "SPECIAL_CATEGORY"


def test_profile_links_and_us_phones_in_text():
    text = "Reach Tiburce at https://www.instagram.com/tiburce-evans, 001-691-518-9820x5621 or (223) 765-4321."
    found = pii.scan_text(text).entities
    assert found["PROFILE_URL"] == 1 and found["PHONE"] == 2
    masked = pii.mask_text(text)
    assert "tiburce-evans" not in masked and "765-4321" not in masked and masked.count(",") == 1
    # IDs, dates and site links are left alone
    assert not pii.scan_text("order 1234567890, id 2021-03-1234, ticket 100-200-3000, https://t.co/abc123").entities


def test_long_health_notes_stay_free_text():
    notes = [f"SUBJECTIVE: patient {i} presents with a cough and mild fever for three days." for i in range(20)]
    _, v = issues(pd.DataFrame({"transcription": notes}))
    p = v.profiles["transcription"]
    assert p.pii_entity == "HEALTH" and p.free_text


def test_hygiene_issues():
    n = 60
    df = pd.DataFrame({
        "Unnamed: 0": [str(i) for i in range(n)],
        "Name": ["Bobby JacksOn" if i < 10 else "Leslie Terry" for i in range(n)] ,
        "product_name": ["Air Cooler??????(White)" if i < 5 else "Air Cooler (White)" for i in range(n)],
        "Ticket Description": [f"My {{product_purchased}} stopped working after update {i}, please help me." for i in range(n)],
        "Billing Amount": [str(-120.5) if i < 3 else str(1000 + i) for i in range(n)],
        "Age": [str(250) if i == 0 else str(30 + i % 40) for i in range(n)],
        "city": ["Pune"] * n,
    })
    found, _ = issues(df)
    assert found["export_index"] == 1
    assert found["name_case"] == 10
    assert found["garbled_text"] == 5
    assert found["template_placeholder"] == n
    assert found["negative_amount"] == 3
    assert found["impossible_age"] == 1


def test_mcdonald_and_tweet_question_marks_are_fine():
    n = 30
    df = pd.DataFrame({
        "full_name": ["Ronald McDonald", "Anil DeSouza", "Mary O'Brien"] * 10,
        "tweet": [f"Did you see the game last night??? Unbelievable finish number {i}!" for i in range(n)],
    })
    found, _ = issues(df)
    assert "name_case" not in found and "garbled_text" not in found


def test_aadhaar_in_prose_needs_the_word():
    # 2345 6789 0123-style numbers pass Verhoeff 1 time in 10; only count them when introduced as Aadhaar
    valid = next(f"2345678901{a}{b}" for a in range(10) for b in range(10) if pii.is_aadhaar(f"2345678901{a}{b}"))
    assert pii.scan_text(f"Customer shared Aadhaar {valid} for KYC").entities["AADHAAR"] == 1
    assert "AADHAAR" not in pii.scan_text(f"My student ID number is {valid}.").entities


def test_serialized_lists_are_not_prose():
    tokens = [str(["Tiburce", "Evans", ",", "https://www.instagram.com/x", str(i)]) for i in range(20)]
    _, v = issues(pd.DataFrame({"tokens": tokens}))
    assert not v.profiles["tokens"].free_text


def test_survey_answers_are_not_duplicates():
    from app.scanner.rows import duplicate_key_columns

    df = pd.DataFrame({"Sector": ["Private", "Government"] * 50, "Degree": ["Yes", "No"] * 50,
                       "Workdays": ["5", "4"] * 50})
    _, v = issues(df)
    assert duplicate_key_columns(v)[0] == []


def test_windows_encoded_csv_reads_cleanly(tmp_path):
    from app.scanner.readers import read_table

    p = tmp_path / "sms.csv"
    p.write_bytes("﻿v1,v2\nspam,Win £1000 cash now\n".encode("utf-8-sig"))
    assert list(read_table(p, 100).columns) == ["v1", "v2"]
    p.write_bytes("v1,v2\nspam,Win £1000 cash now\n".encode("cp1252"))
    assert read_table(p, 100)["v2"][0] == "Win £1000 cash now"
