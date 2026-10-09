"""Generate the synthetic demo estates (Acme Retail, Zenith Finance).

Every problem is planted on purpose so each screen has something real to show. All personal
data is fake: random Indian-style names with checksum-valid but randomly generated Aadhaar/PAN
numbers. Never put real personal data in the demo.

    python -m scripts.generate_demo            # writes backend/demo/
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import sqlite3
import string
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.scanner.pii import verhoeff_check_digit  # noqa: E402

VERSION = "4"  # bump to force regeneration
NOW = datetime.now().replace(microsecond=0)

FIRST = ["Rahul", "Priya", "Amit", "Sneha", "Vikram", "Ananya", "Rohan", "Kavya", "Arjun", "Isha", "Karan", "Meera",
         "Siddharth", "Pooja", "Aditya", "Neha", "Varun", "Divya", "Nikhil", "Shreya", "Manish", "Ritu", "Suresh",
         "Lakshmi", "Harsh", "Tanvi", "Gaurav", "Aisha", "Deepak", "Nandini", "Rajesh", "Swati", "Abhishek", "Kriti",
         "Mohit", "Pallavi", "Yash", "Anjali", "Sanjay", "Bhavna", "Imran", "Farah", "Joseph", "Mary", "Gurpreet",
         "Harpreet", "Venkat", "Lavanya", "Arvind", "Shalini"]
LAST = ["Sharma", "Verma", "Patel", "Iyer", "Reddy", "Nair", "Gupta", "Singh", "Kumar", "Das", "Mehta", "Joshi",
        "Rao", "Menon", "Chopra", "Kapoor", "Bose", "Mukherjee", "Agarwal", "Pillai", "Khan", "Fernandes", "Gill",
        "Naidu", "Desai", "Kulkarni", "Banerjee", "Saxena", "Malhotra", "Chauhan"]
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
          "Kochi", "Indore", "Chandigarh", "Nagpur", "Surat", "Bhopal", "Coimbatore", "Vizag", "Noida", "Gurugram"]
DOMAINS = ["gmail.com", "yahoo.co.in", "outlook.com", "rediffmail.com", "hotmail.com"]


def ago(days: float) -> datetime:
    return NOW - timedelta(days=days)


class Faker:
    def __init__(self, seed: int):
        self.r = random.Random(seed)

    def aadhaar(self) -> str:
        base = str(self.r.randint(2, 9)) + "".join(self.r.choice(string.digits) for _ in range(10))
        return base + verhoeff_check_digit(base)

    def pan(self, last: str) -> str:
        return ("".join(self.r.choice(string.ascii_uppercase) for _ in range(3)) + "P" + last[0].upper()
                + "".join(self.r.choice(string.digits) for _ in range(4)) + self.r.choice(string.ascii_uppercase))

    def phone(self) -> str:
        return self.r.choice("6789") + "".join(self.r.choice(string.digits) for _ in range(9))

    def person(self, i: int) -> dict:
        f, l = self.r.choice(FIRST), self.r.choice(LAST)
        return {
            "first": f, "last": l, "name": f"{f} {l}",
            "email": f"{f.lower()}.{l.lower()}{i}@{self.r.choice(DOMAINS)}",
            "phone": self.phone(), "pan": self.pan(l), "aadhaar": self.aadhaar(),
            "city": self.r.choice(CITIES),
            "dob": (NOW - timedelta(days=self.r.randint(20 * 365, 65 * 365))).date(),
            "signup": (NOW - timedelta(days=self.r.randint(30, 4 * 365))).date(),
        }

    def phone_variant(self, p: str) -> str:
        return self.r.choice([f"+91{p}", f"+91 {p[:5]} {p[5:]}", f"0{p}", p, f"{p[:5]}-{p[5:]}", f"91{p}", f"+91-{p}"])

    def date_variant(self, d) -> str:
        k = self.r.random()
        if k < 0.65:
            return d.strftime("%Y-%m-%d")
        if k < 0.85:
            return d.strftime("%d/%m/%Y")
        return d.strftime("%d %b %Y")


# ------------------------------------------------------------------ file writers


def touch(path: Path, when: datetime) -> None:
    ts = when.timestamp()
    os.utime(path, (ts, ts))


def write_pdf(path: Path, title: str, lines: list[str]) -> None:
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    text = ["BT", "/F1 16 Tf", "50 750 Td", "18 TL", f"({esc(title)}) Tj", "/F1 10 Tf", "14 TL", "T*"]
    text += [f"({esc(l)}) '" for l in lines[:48]]
    text.append("ET")
    stream = "\n".join(text).encode("latin-1", "replace")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def write_docx(path: Path, paragraphs: list[str]) -> None:
    body = "".join(f"<w:p><w:r><w:t xml:space=\"preserve\">{p.replace('&', '&amp;').replace('<', '&lt;')}</w:t></w:r></w:p>"
                   for p in paragraphs)
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           f"<w:body>{body}</w:body></w:document>")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        z.writestr("word/document.xml", doc)


# ------------------------------------------------------------------ ACME RETAIL


def acme(root: Path) -> None:
    fk = Faker(42)
    files = root / "files"
    manifest: dict[str, dict] = {}

    def reg(rel: str, *, owner: str, dept: str, shared: int, modified_days: float, accessed_days: float | None = None,
            encrypted: bool = False):
        p = files / rel
        touch(p, ago(modified_days))
        manifest[rel] = {"owner": owner, "department": dept, "shared_with": shared, "encrypted": encrypted,
                         "modified_at": ago(modified_days).isoformat(),
                         "last_accessed": ago(accessed_days if accessed_days is not None else modified_days).isoformat()}

    for d in ("crm", "exports", "reports", "backups", "marketing", "finance", "hr", "legacy", "config", "docs",
              "Downloads", "databases"):
        (files / d).mkdir(parents=True, exist_ok=True)

    # --- crm/customers_master.csv: quality issues + exact & fuzzy duplicates + PAN conflicts
    people = [fk.person(i) for i in range(7600)]
    rows = []
    for i, p in enumerate(people):
        rows.append({
            "customer_id": f"C{100000 + i}", "full_name": p["name"], "email": p["email"], "phone": p["phone"],
            "city": p["city"], "signup_date": p["signup"], "pan": p["pan"] if fk.r.random() < 0.6 else "",
            "loyalty_tier": fk.r.choice(["Silver", "Gold", "Platinum", "Silver", "Silver"]),
            "age": fk.r.randint(19, 72),
            "lifetime_value_inr": int(fk.r.lognormvariate(9.6, 0.7)),
        })
    for k, (age, ltv) in zip(range(400, 409), [(250, None), (999, None), (-4, None), (None, 98_500_000), (None, 45_000_000),
                                                (None, 120_000_000), (187, None), (None, 76_000_000), (0, None)]):
        if age is not None:
            rows[k]["age"] = age
        if ltv is not None:
            rows[k]["lifetime_value_inr"] = ltv
    rows[0].update(full_name="Rahul", email="rahul@gmail.com", phone="9876543210", pan="")
    dup_rows = [
        {**rows[0], "customer_id": "C900101", "full_name": "Rahul Sharma"},
        {**rows[0], "customer_id": "C900102"},
    ]
    for k in range(1, 121):  # exact duplicates (re-imported rows)
        dup_rows.append(dict(rows[k]))
    for k in range(121, 301):  # same person, different spelling / formatting
        src = rows[k]
        first, last = src["full_name"].split(" ", 1) if " " in src["full_name"] else (src["full_name"], "")
        variant = fk.r.choice([first, f"{first} {last[0]}." if last else first, f"{first.upper()} {last}", f"{last} {first}"])
        dup_rows.append({**src, "customer_id": f"C9{k:05d}", "full_name": variant,
                         "phone": src["phone"] if fk.r.random() < 0.7 else ""})
    for k in range(301, 316):  # same email, different PAN -> must NOT be merged
        src = rows[k]
        dup_rows.append({**src, "customer_id": f"C8{k:05d}", "pan": fk.pan(src["full_name"].split()[-1]),
                         "full_name": src["full_name"]})
        if not src["pan"]:
            src["pan"] = fk.pan(src["full_name"].split()[-1])
    all_rows = rows + dup_rows
    fk.r.shuffle(all_rows)
    for r in all_rows:  # formatting noise
        if fk.r.random() < 0.06:
            r["full_name"] = "  " + r["full_name"] + " "
        if fk.r.random() < 0.05 and r["email"]:
            r["email"] = r["email"].upper() if fk.r.random() < 0.5 else r["email"].capitalize()
        if r["phone"] and fk.r.random() < 0.35:
            r["phone"] = fk.phone_variant(r["phone"])
        r["signup_date"] = fk.date_variant(r["signup_date"])
        if fk.r.random() < 0.02:
            r["email"] = ""
        elif fk.r.random() < 0.004:
            r["email"] = r["email"].replace("@", " at ")
    pd.DataFrame(all_rows).to_csv(files / "crm/customers_master.csv", index=False)
    reg("crm/customers_master.csv", owner="neha.joshi@acme.in", dept="CRM", shared=24, modified_days=6, accessed_days=1)
    shutil.copy(files / "crm/customers_master.csv", files / "Downloads/customers_master (1).csv")
    reg("Downloads/customers_master (1).csv", owner="intern.sales@acme.in", dept="Sales", shared=3, modified_days=40)

    # --- exports/customer_data_*: near-duplicate dataset versions
    clean = pd.DataFrame([{"name": p["name"], "email": p["email"], "phone": "+91" + p["phone"], "city": p["city"]}
                          for p in people[:2000]])
    clean.to_csv(files / "exports/customer_data_latest.csv", index=False)
    clean.sample(frac=1.0, random_state=7).to_csv(files / "exports/customer_data_final2.csv", index=False)
    clean.iloc[:1950].to_csv(files / "exports/customer_data_final.csv", index=False)
    clean.iloc[:1800].to_csv(files / "exports/customer_data_2024.csv", index=False)
    zoho = clean.iloc[40:2000].rename(columns={"name": "Customer Name", "email": "E-mail Address", "phone": "Contact Number", "city": "Town"})
    zoho.sample(frac=1.0, random_state=11).to_csv(files / "exports/zoho_crm_export.csv", index=False)
    reg("exports/zoho_crm_export.csv", owner="sales.ops@acme.in", dept="Sales", shared=6, modified_days=210, accessed_days=200)
    for name, d in [("customer_data_2024.csv", 300), ("customer_data_final.csv", 120), ("customer_data_final2.csv", 95),
                    ("customer_data_latest.csv", 30)]:
        reg(f"exports/{name}", owner="analytics@acme.in", dept="Analytics", shared=4, modified_days=d, accessed_days=d - 5)

    # --- reports/Report*.xlsx: the classic version chain
    regions, months = ["North", "South", "East", "West", "Central"], pd.period_range("2024-01", periods=24, freq="M")
    base = [{"region": rg, "month": str(m), "revenue_inr": fk.r.randint(800_000, 5_000_000),
             "orders": fk.r.randint(900, 6000), "returns": fk.r.randint(10, 300)} for m in months for rg in regions]
    chain = [("Report.xlsx", 60, 210), ("Report_v2.xlsx", 80, 180), ("Report_final.xlsx", 100, 150),
             ("Report_final2.xlsx", 110, 120), ("Report_FINAL_REAL.xlsx", 120, 90), ("Report_FINAL_REAL_NEW.xlsx", 120, 85)]
    for name, n, d in chain:
        df = pd.DataFrame(base[:n])
        if name == "Report_final2.xlsx":
            df.loc[3, "revenue_inr"] += 1500
        df.to_excel(files / f"reports/{name}", index=False)
        reg(f"reports/{name}", owner="cfo.office@acme.in", dept="Finance", shared=9, modified_days=d)

    # --- backups/orders_backup_01..05: 01 obsolete, 02 99.9% identical, 03 identical, 04 useful, 05 latest
    orders = pd.DataFrame([{"order_id": f"O{500000 + i}", "customer_id": f"C{100000 + fk.r.randint(0, 7599)}",
                            "amount_inr": round(fk.r.uniform(199, 25000), 2), "status": fk.r.choice(["delivered", "shipped", "returned"]),
                            "order_date": (NOW - timedelta(days=fk.r.randint(1, 700))).date().isoformat()} for i in range(3000)])
    b05 = orders
    b04 = orders.iloc[:2940]
    b02 = b04.copy()
    b02.loc[17, "status"] = "returned" if b02.loc[17, "status"] != "returned" else "delivered"
    b01 = orders.iloc[:2400]
    for name, df, d in [("orders_backup_01.csv", b01, 150), ("orders_backup_02.csv", b02, 120),
                        ("orders_backup_03.csv", b04, 90), ("orders_backup_04.csv", b04, 60), ("orders_backup_05.csv", b05, 30)]:
        df.to_csv(files / f"backups/{name}", index=False)
        reg(f"backups/{name}", owner="dba@acme.in", dept="IT", shared=3, modified_days=d)

    # --- marketing/marketing_old.zip: big, cold, 2019 campaign assets
    zr = random.Random(9)
    with zipfile.ZipFile(files / "marketing/marketing_old.zip", "w", zipfile.ZIP_STORED) as z:
        for i in range(24):
            zi = zipfile.ZipInfo(f"campaign_2019/diwali_banner_{i:02d}.png", date_time=(2019, 10, 1 + i % 28, 10, 0, 0))
            z.writestr(zi, zr.randbytes(220_000))
        for i in range(3):
            zi = zipfile.ZipInfo(f"campaign_2019/tv_spot_cut{i + 1}.mov", date_time=(2019, 11, 5, 12, 0, 0))
            z.writestr(zi, zr.randbytes(9_000_000))
    reg("marketing/marketing_old.zip", owner="brand@acme.in", dept="Marketing", shared=2, modified_days=1700, accessed_days=1460)

    # --- hr/employee_records.csv: Aadhaar + PAN + bank, over-shared
    emp = []
    for i in range(640):
        p = fk.person(10_000 + i)
        emp.append({"employee_id": f"E{2000 + i}", "full_name": p["name"], "email": f"{p['first'].lower()}.{p['last'].lower()}@acme.in",
                    "phone": "+91" + p["phone"], "aadhaar": p["aadhaar"], "pan": p["pan"],
                    "bank_account_no": "".join(fk.r.choice(string.digits) for _ in range(12)),
                    "ifsc": fk.r.choice(["HDFC", "ICIC", "SBIN", "UTIB"]) + "0" + "".join(fk.r.choice(string.digits) for _ in range(6)),
                    "date_of_birth": p["dob"].isoformat(), "monthly_salary_inr": fk.r.randint(25_000, 400_000)})
    for k, sal in zip((17, 233, 401), (45_000_000, 38_500_000, 9)):
        emp[k]["monthly_salary_inr"] = sal  # a crore-scale typo and a 9-rupee salary
    pd.DataFrame(emp).to_csv(files / "hr/employee_records.csv", index=False)
    reg("hr/employee_records.csv", owner="hr.ops@acme.in", dept="HR", shared=37, modified_days=20, accessed_days=2)

    # --- legacy/customer_backup_2023.csv: Aadhaar KYC data unused for ~3.7 years (DPDP erasure candidate)
    legacy = []
    for i in range(45_000):
        p = fk.person(50_000 + i)
        legacy.append({"cust_ref": f"L{i:06d}", "name": p["name"], "mobile": p["phone"], "email": p["email"],
                       "aadhaar_no": p["aadhaar"], "kyc_city": p["city"], "kyc_date": p["signup"].isoformat()})
    pd.DataFrame(legacy).to_csv(files / "legacy/customer_backup_2023.csv", index=False)
    reg("legacy/customer_backup_2023.csv", owner="old.it.vendor@acme.in", dept="IT", shared=14, modified_days=1500, accessed_days=1350)

    # --- finance/vendor_payments.xlsx: PAN + bank accounts
    vend = [{"vendor": f"{fk.r.choice(LAST)} {fk.r.choice(['Traders', 'Logistics', 'Packaging', 'Foods'])}",
             "pan": fk.pan(fk.r.choice(LAST)), "account_number": "".join(fk.r.choice(string.digits) for _ in range(14)),
             "ifsc": "HDFC0" + "".join(fk.r.choice(string.digits) for _ in range(6)),
             "amount_inr": fk.r.randint(10_000, 900_000)} for _ in range(220)]
    pd.DataFrame(vend).to_excel(files / "finance/vendor_payments_FY24.xlsx", index=False)
    reg("finance/vendor_payments_FY24.xlsx", owner="accounts@acme.in", dept="Finance", shared=6, modified_days=200, encrypted=True)

    # --- config/.env.bak: leaked secrets
    (files / "config/.env.bak").write_text(
        "# production settings - DO NOT SHARE\nAWS_ACCESS_KEY_ID=AKIA" + "".join(random.Random(3).choice(string.ascii_uppercase + string.digits) for _ in range(16))
        + "\nAWS_SECRET_ACCESS_KEY=" + "".join(random.Random(4).choice(string.ascii_letters + string.digits) for _ in range(40))
        + "\nDB_PASSWORD=Acme@Prod#2023\nRAZORPAY_KEY=rzp_live_" + "".join(random.Random(5).choice(string.ascii_letters + string.digits) for _ in range(14)) + "\n")
    reg("config/.env.bak", owner="devops@acme.in", dept="IT", shared=18, modified_days=400)

    # --- docs: version chain (text similarity), exact duplicate, notes with an Aadhaar
    policy = [f"Section {i}: Customers must complete KYC verification before high-value orders above Rs 50,000. "
              f"Documents are retained for the period required by law and reviewed every quarter." for i in range(1, 30)]
    write_pdf(files / "docs/KYC_Policy_v1.pdf", "Acme KYC Policy v1", policy[:22])
    write_pdf(files / "docs/KYC_Policy_final.pdf", "Acme KYC Policy", policy[:26])
    write_pdf(files / "docs/KYC_Policy_final_v2.pdf", "Acme KYC Policy", policy[:29])
    for name, d in [("KYC_Policy_v1.pdf", 500), ("KYC_Policy_final.pdf", 380), ("KYC_Policy_final_v2.pdf", 200)]:
        reg(f"docs/{name}", owner="compliance@acme.in", dept="Legal", shared=40, modified_days=d)
    retention = [
        "Customer personal data must be deleted within thirty days after the account is closed.",
        "Backups that contain personal data are kept for no longer than ninety days.",
        "Access to customer records is limited to staff who need it for their job.",
        "Aadhaar and PAN numbers must always be stored encrypted and masked in reports.",
        "Every deletion request from a customer has to be completed within one week and logged.",
        "Marketing teams may not export customer lists to personal devices or email.",
        "Data owners review who has access to sensitive files every quarter.",
        "Log files with personal information are removed after one year.",
        "Vendors that process customer data must sign a data processing agreement first.",
        "Any suspected data breach is reported to the security team within one hour.",
    ] * 2
    reworded = [
        "Personal information about customers shall be erased no later than 30 days once their account has been closed.",
        "Backup copies holding personal information are retained for a maximum of 90 days.",
        "Only employees whose role requires it can access customer records.",
        "PAN and Aadhaar numbers are to be kept in encrypted form and masked whenever they appear in reports.",
        "Each customer request to delete data needs to be finished inside seven days and recorded.",
        "The marketing department is not allowed to copy customer lists onto personal laptops, phones or email.",
        "Owners of data check the access list for sensitive files once every three months.",
        "Logs that include personal information get deleted after twelve months.",
        "Suppliers handling customer information have to sign a data processing agreement before starting.",
        "Possible data breaches must be escalated to the security team in under sixty minutes.",
    ] * 2
    write_docx(files / "docs/Data_Retention_Policy_2022.docx", ["Acme Data Retention Policy (2022)"] + retention)
    (files / "Downloads").mkdir(exist_ok=True)
    write_pdf(files / "legacy/Retention_Guidelines_v3.pdf", "Retention guidelines", reworded)
    reg("docs/Data_Retention_Policy_2022.docx", owner="compliance@acme.in", dept="Legal", shared=40, modified_days=720)
    reg("legacy/Retention_Guidelines_v3.pdf", owner="old.it.vendor@acme.in", dept="IT", shared=9, modified_days=400)
    write_docx(files / "docs/Vendor_Contract_Template.docx",
               [f"Clause {i}: The vendor shall deliver goods as per the purchase order and agreed service levels." for i in range(1, 40)])
    shutil.copy(files / "docs/Vendor_Contract_Template.docx", files / "Downloads/Vendor_Contract_Template copy.docx")
    reg("docs/Vendor_Contract_Template.docx", owner="legal@acme.in", dept="Legal", shared=12, modified_days=300)
    reg("Downloads/Vendor_Contract_Template copy.docx", owner="intern.sales@acme.in", dept="Sales", shared=2, modified_days=250)
    a = people[5]
    (files / "docs/support_call_notes.txt").write_text(
        f"Call with customer {a['name']} on refund.\nCustomer shared Aadhaar {a['aadhaar'][:4]} {a['aadhaar'][4:8]} {a['aadhaar'][8:]} "
        f"and PAN {a['pan']} over chat for verification.\nCallback number {a['phone']}.\nResolved: refund issued.\n")
    reg("docs/support_call_notes.txt", owner="support@acme.in", dept="Support", shared=22, modified_days=45)

    # --- databases/crm.db + query log: unused columns
    db_path = files / "databases/crm.db"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT, phone TEXT, address TEXT, city TEXT,"
                " age INTEGER, fax_number TEXT, middle_name TEXT, old_customer_code TEXT, created_at TEXT)")
    con.execute("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL, status TEXT,"
                " created_at TEXT, legacy_discount_code TEXT)")
    custs = []
    for i in range(5000):
        p = people[i]
        custs.append((i + 1, p["name"], p["email"], "+91" + p["phone"], f"{fk.r.randint(1, 999)}, {fk.r.choice(['MG Road', 'Park Street', 'Linking Road', 'Anna Salai'])}",
                      p["city"], fk.r.randint(19, 70), ("022-" + str(fk.r.randint(2000000, 2999999))) if fk.r.random() < 0.3 else None,
                      fk.r.choice(["Kumar", "Lal", "Prasad", None, None, None]), f"OLD-{fk.r.randint(10000, 99999)}" if fk.r.random() < 0.8 else None,
                      (NOW - timedelta(days=fk.r.randint(1, 2000))).isoformat()))
    con.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?,?,?,?,?)", custs)
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", [
        (i + 1, fk.r.randint(1, 5000), round(fk.r.uniform(99, 20000), 2), fk.r.choice(["paid", "refunded", "pending"]),
         (NOW - timedelta(days=fk.r.randint(1, 1500))).isoformat(), f"DISC{fk.r.randint(1, 99)}" if fk.r.random() < 0.2 else None)
        for i in range(20000)])
    con.commit()
    con.close()
    reg("databases/crm.db", owner="dba@acme.in", dept="IT", shared=6, modified_days=1, accessed_days=0)

    conn_dir = root / "connectors"
    conn_dir.mkdir(parents=True, exist_ok=True)
    qr = random.Random(11)
    recent = [
        ("crm-web", "SELECT name, email, phone FROM customers WHERE id = ?"),
        ("crm-web", "UPDATE customers SET email = ?, phone = ? WHERE id = ?"),
        ("support-portal", "SELECT id, name, address, city FROM customers WHERE phone = ?"),
        ("analytics-dbt", "SELECT city, age, created_at FROM customers"),
        ("billing-service", "SELECT order_id, customer_id, amount, status, created_at FROM orders WHERE customer_id = ?"),
        ("billing-service", "INSERT INTO orders (customer_id, amount, status, created_at) VALUES (?, ?, ?, ?)"),
        ("analytics-dbt", "SELECT o.amount, o.status, c.city FROM orders o JOIN customers c ON c.id = o.customer_id"),
    ]
    log = []
    for _ in range(3200):
        sysname, q = qr.choice(recent)
        log.append((ago(qr.uniform(0, 1460)), sysname, q))
    for _ in range(40):  # legacy ERP stopped ~3 years ago
        log.append((ago(qr.uniform(1100, 1460)), "legacy-erp", "SELECT id, fax_number, old_customer_code FROM customers WHERE id = ?"))
    for _ in range(15):
        log.append((ago(qr.uniform(1300, 1460)), "legacy-erp", "SELECT * FROM customers"))
    for _ in range(25):
        log.append((ago(qr.uniform(700, 900)), "promo-engine", "SELECT order_id, legacy_discount_code FROM orders WHERE created_at > ?"))
    log.sort()
    with (conn_dir / "query_log.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "system", "query"])
        for ts, s, q in log:
            w.writerow([ts.isoformat(), s, q])

    # --- S3 inventory (metadata only): the big numbers
    ir = random.Random(21)
    GB = 1024**3
    inv = []

    def obj(bucket, key, size, modified_days, accessed_days, etag=None, owner="data-eng@acme.in", dept="Data", shared=5,
            sc="STANDARD", enc="SSE-S3"):
        inv.append({"bucket": bucket, "key": key, "size": int(size), "last_modified": ago(modified_days).isoformat(),
                    "etag": etag or hashlib.md5(f"{bucket}/{key}/{size}".encode()).hexdigest(), "storage_class": sc,
                    "last_accessed": ago(accessed_days).isoformat(), "owner": owner, "department": dept,
                    "shared_with": shared, "encrypted": enc})

    for year, stale_days in [(2021, 1200), (2022, 520), (2023, None), (2024, None), (2025, None)]:
        for m in range(1, 13):
            for part in range(20):
                key = f"events/year={year}/month={m:02d}/part-{part:04d}.parquet"
                size = ir.uniform(1.9, 2.6) * GB
                mod = (NOW.year - year) * 365 + (12 - m) * 30
                acc = stale_days + ir.uniform(0, 60) if stale_days else ir.uniform(0, 90)
                obj("acme-prod-data", key, size, mod, min(acc, mod) if stale_days is None else acc)
    snapshots = []
    for k in range(24):
        d = ago(30 * k + 2)
        key = f"postgres/prod_snapshot_{d:%Y-%m-%d}.dump"
        size = ir.uniform(40, 52) * GB
        et = hashlib.md5(key.encode()).hexdigest()
        snapshots.append((key, size, et, 30 * k + 2))
        obj("acme-backups", key, size, 30 * k + 2, 30 * k + 2, etag=et, owner="dba@acme.in", dept="IT", shared=3)
    for key, size, et, d in snapshots[2:5]:  # manual copies left in the data bucket
        obj("acme-prod-data", f"tmp/db_export_{key.split('_')[-1].replace('.dump', '')}.dump", size, d - 1, d - 1, etag=et,
            owner="ravi.dev@acme.in", dept="Engineering", shared=11)
    for i in range(40):
        yr = ir.choice([2019, 2020, 2021])
        obj("acme-marketing", f"campaigns/{yr}/campaign_{yr}_{i:02d}_master.mp4", ir.uniform(5.2, 24) * GB,
            (NOW.year - yr) * 365, (NOW.year - yr) * 365 - ir.uniform(0, 120), owner="brand@acme.in", dept="Marketing", shared=2)
    brand_et = hashlib.md5(b"brand").hexdigest()
    obj("acme-marketing", "final/brand_film_2025.mp4", 12.4 * GB, 200, 15, etag=brand_et, owner="brand@acme.in", dept="Marketing", shared=12)
    obj("acme-marketing", "shared/brand_film_2025_copy.mp4", 12.4 * GB, 150, 150, etag=brand_et, owner="agency@partner.in", dept="Marketing", shared=12)
    for i, (name, d) in enumerate([("Investor_Deck_v1.pptx", 400), ("Investor_Deck_v2.pptx", 330),
                                   ("Investor_Deck_final.pptx", 300), ("Investor_Deck_final_v2.pptx", 280)]):
        obj("acme-corp-docs", f"board/{name}", ir.uniform(140, 180) * 1024**2, d, d - 10, owner="ceo.office@acme.in", dept="Leadership", shared=6)
    for i in range(30):
        d = ir.uniform(20, 900)
        obj("acme-ml", f"models/recsys/checkpoint_{i:03d}.pt", ir.uniform(2.5, 3.5) * GB, d, d - 5, owner="ml@acme.in", dept="Data Science", shared=4)
    with (conn_dir / "s3_inventory.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(inv[0].keys()))
        w.writeheader()
        w.writerows(inv)

    (conn_dir / "access_manifest.json").write_text(json.dumps(
        {"source": "Google Drive sharing export + IAM (simulated)",
         "defaults": {"owner": "unknown", "shared_with": 1, "encrypted": False},
         "files": manifest}, indent=1))


# ------------------------------------------------------------------ ZENITH FINANCE (second tenant)


def zenith(root: Path) -> None:
    fk = Faker(77)
    files = root / "files"
    (files / "loans").mkdir(parents=True, exist_ok=True)
    (files / "kyc").mkdir(parents=True, exist_ok=True)
    manifest = {}
    apps = []
    for i in range(2400):
        p = fk.person(i)
        apps.append({"applicant_id": f"Z{70000 + i}", "full_name": p["name"], "pan": p["pan"], "aadhaar": p["aadhaar"],
                     "email": p["email"], "phone": fk.phone_variant(p["phone"]), "loan_amount_inr": fk.r.randint(50_000, 2_500_000),
                     "status": fk.r.choice(["approved", "rejected", "pending", "disbursed"]), "applied_on": fk.date_variant(p["signup"])})
    extra = []
    for k in range(60):  # re-applications with the same PAN
        extra.append({**apps[k], "applicant_id": f"Z9{k:04d}", "email": apps[k]["email"].upper()})
    for k in range(60, 72):  # same PAN, different Aadhaar -> conflict
        extra.append({**apps[k], "applicant_id": f"Z8{k:04d}", "aadhaar": fk.aadhaar()})
    for k, amt in zip((5, 77, 1300, 2001), (950_000_000, 1, 480_000_000, 0)):
        apps[k]["loan_amount_inr"] = amt
    rows = apps + extra
    fk.r.shuffle(rows)
    pd.DataFrame(rows).to_csv(files / "loans/loan_applicants.csv", index=False)
    manifest["loans/loan_applicants.csv"] = {"owner": "credit.ops@zenith.in", "department": "Credit", "shared_with": 15,
                                             "modified_at": ago(3).isoformat(), "last_accessed": ago(1).isoformat()}
    write_pdf(files / "kyc/KYC_Checklist.pdf", "Zenith KYC Checklist",
              [f"Step {i}: verify PAN and Aadhaar of the applicant before disbursal." for i in range(1, 20)])
    manifest["kyc/KYC_Checklist.pdf"] = {"owner": "compliance@zenith.in", "department": "Compliance", "shared_with": 30,
                                         "modified_at": ago(100).isoformat(), "last_accessed": ago(10).isoformat()}
    for rel, m in manifest.items():
        touch(files / rel, datetime.fromisoformat(m["modified_at"]))
    conn = root / "connectors"
    conn.mkdir(parents=True, exist_ok=True)
    (conn / "access_manifest.json").write_text(json.dumps({"defaults": {"shared_with": 1}, "files": manifest}, indent=1))


def ensure_demo(demo_dir: Path, force: bool = False) -> None:
    marker = demo_dir / ".version"
    if not force and marker.exists() and marker.read_text().strip() == f"{VERSION}:{NOW.date()}":
        return
    shutil.rmtree(demo_dir, ignore_errors=True)
    acme(demo_dir / "acme")
    zenith(demo_dir / "zenith")
    marker.write_text(f"{VERSION}:{NOW.date()}")


if __name__ == "__main__":
    from app.config import get_settings

    d = get_settings().demo_dir
    ensure_demo(d, force=True)
    total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    print(f"demo estates written to {d} ({total / 1024**2:.1f} MB on disk)")
