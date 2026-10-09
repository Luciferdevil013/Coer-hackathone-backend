"""Anomalous-value detection: Isolation Forest + robust z-scores.

Isolation Forest (scikit-learn) finds rows that are unusual across all numeric columns
together; the robust z-score (median / MAD) says WHICH value is unusual and by how much.
A row is reported only when both agree, plus explicit rules for amounts 100x above the median
or zero / 1000x below it (a ₹0 loan, a ₹9 salary). That keeps precision high and every flag explainable
("age = 250 is 9.9 robust deviations above the median of 45").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_ROWS = 200
_ID_LIKE = re.compile(r"(^|_)(id|no|num|number|code|pin|pincode|zip|year|phone|mobile|account)($|_)")
_SENSITIVE = {"PHONE", "AADHAAR", "CREDIT_CARD", "BANK_ACCOUNT", "PAN"}


@dataclass
class AnomalyResult:
    columns: list[str]
    rows: list[dict] = field(default_factory=list)   # [{row, score, flags:[{column, value, median, z}]}]
    n_rows: int = 0
    params: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.rows)


def _to_number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def numeric_columns(df: pd.DataFrame, profiles: dict | None = None) -> list[str]:
    """Columns that are genuinely measured quantities (not IDs, codes, phone numbers...)."""
    out = []
    for c in df.columns:
        norm = re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")
        p = (profiles or {}).get(c)
        if p is not None and (p.inferred_type not in ("integer", "decimal") or p.pii_entity in _SENSITIVE):
            continue
        if _ID_LIKE.search(norm) or re.search(r"[a-z](ID|Id)$", str(c)):   # DeptID, ManagerID: codes, not quantities
            continue
        vals = _to_number(df[c]).dropna()
        if len(vals) < MIN_ROWS or vals.nunique() <= 5:
            continue
        if vals.nunique() <= 10 and bool((vals == vals.round()).all()):
            continue  # small counts and ratings (days late 0-6, a 1-5 score): a 6 is uncommon, not wrong
        if vals.nunique() > 0.98 * len(vals) and float(np.median(np.diff(np.sort(vals.to_numpy())))) == 1.0:
            continue  # sequential identifiers (1, 2, 3, ...), not measurements
        out.append(c)
    return out


_MONEY = re.compile(r"(inr|rs|rupee|amount|amt|salary|price|revenue|spend|cost|balance|fee|income|loan|limit|ltv|billing)")


def _transform(x: pd.Series, name: str = "") -> tuple[pd.Series, bool]:
    """Money and counts are right-skewed: judge them on a log scale (ratios), so a big-but-normal
    customer is not an 'anomaly' while a ₹9.8-crore typo or a ₹9 salary still is."""
    v = x.dropna()
    if not len(v) or (v < 0).any():
        return x, False
    lv = np.log10(1 + v)
    # use the log scale only if it makes the bulk of the data MORE symmetric (lognormal-like);
    # quartile (Bowley) skew is immune to the very outliers we are hunting
    return (np.log10(1 + x), True) if abs(_bowley(lv)) < abs(_bowley(v)) - 0.05 else (x, False)


def _bowley(v: pd.Series) -> float:
    q1, q2, q3 = (float(v.quantile(q)) for q in (0.25, 0.5, 0.75))
    return (q3 + q1 - 2 * q2) / (q3 - q1) if q3 > q1 else 0.0


def is_money(name: str) -> bool:
    return bool(_MONEY.search(re.sub(r"[^a-z0-9]+", "_", name.lower())))


def effective_threshold(z_threshold: float, n: int) -> float:
    """Bonferroni-style: with n rows, require a z whose two-sided normal tail expects < 0.5 false
    positives. 640 rows -> 3.5 (the floor); 7,917 rows -> 4.0; 100k rows -> 4.6."""
    from statistics import NormalDist

    return max(z_threshold, NormalDist().inv_cdf(1 - 0.25 / max(n, 1)))


def robust_z(x: pd.Series) -> tuple[pd.Series, float]:
    med = float(x.median())
    mad = float((x - med).abs().median())
    if mad > 0:
        return 0.6745 * (x - med) / mad, med
    mean_ad = float((x - x.mean()).abs().mean())
    if mean_ad > 0:  # many identical values: fall back to the mean absolute deviation
        return (x - med) / (1.2533 * mean_ad), med
    return pd.Series(0.0, index=x.index), med


def detect(df: pd.DataFrame, columns: list[str], z_threshold: float = 3.5, random_state: int = 42) -> AnomalyResult:
    from sklearn.ensemble import IsolationForest

    res = AnomalyResult(columns=list(columns), n_rows=len(df),
                        params={"model": "IsolationForest", "n_estimators": 200, "contamination": "auto",
                                "random_state": random_state, "robust_z_threshold": z_threshold,
                                "skewed_columns_on_log_scale": True})
    if not columns or len(df) < MIN_ROWS:
        return res
    X = pd.DataFrame({c: _to_number(df[c]) for c in columns})
    T, zs, meds, logged = {}, {}, {}, {}
    for c in columns:
        T[c], logged[c] = _transform(X[c], c)
        zs[c], _ = robust_z(T[c].dropna())
        meds[c] = float(X[c].median())
    T = pd.DataFrame(T)
    filled = T.fillna(T.median())
    z_threshold = effective_threshold(z_threshold, len(df))
    res.params["robust_z_threshold"] = round(z_threshold, 2)
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=random_state)
    model.fit(filled.to_numpy())
    scores = model.score_samples(filled.to_numpy())     # lower = more anomalous
    outlier = model.predict(filled.to_numpy()) == -1
    money = {c: is_money(c) and meds[c] > 0 for c in columns}
    for pos in range(len(X)):
        flags = []
        for c in columns:
            v = X[c].iat[pos]
            if pd.isna(v):
                continue
            z = float(zs[c].get(X.index[pos], 0.0))
            base = {"column": c, "value": float(v), "median": round(meds[c], 2), "z": round(z, 1),
                    "scale": "log" if logged[c] else "linear"}
            if outlier[pos] and abs(z) >= z_threshold:
                flags.append({**base, "rule": "isolation forest + robust z"})
            elif money[c] and outlier[pos] and v >= 100 * meds[c] and abs(z) >= z_threshold:
                # order-of-magnitude typo in an amount (a ₹9.8-crore lifetime value) - only when the forest agrees,
                # since a catalogue legitimately holds a ₹87k air conditioner next to ₹675 accessories
                flags.append({**base, "rule": "≥100× above median amount"})
            elif money[c] and (v <= 0 or v <= meds[c] / 1000):
                # a ₹0 loan or a ₹9 salary; small-but-plausible amounts (a ₹186 bill) are left alone
                flags.append({**base, "rule": "zero or ≥1000× below median amount"})
        if flags:
            res.rows.append({"row": int(pos), "score": round(float(scores[pos]), 4), "flags": flags})
    res.rows.sort(key=lambda r: (r["score"], -max(abs(f["z"]) for f in r["flags"])))
    return res
