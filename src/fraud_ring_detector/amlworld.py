"""IBM AMLworld (Altman et al., NeurIPS 2023) adapter: schema detection, laundering-pattern parsing, time-window selection.

Raw columns: Timestamp, From Bank, Account, To Bank, Account.1, Amount Received, Receiving Currency, Amount Paid,
Payment Currency, Payment Format, Is Laundering  (pandas renames the second `Account` to `Account.1`).
The *_Patterns.txt file lists each laundering attempt between BEGIN/END markers, one transaction per line.
"""
import csv
import re

import numpy as np
import pandas as pd

AML_COLS = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Received", "Receiving Currency",
            "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]
STR_COLS = {"From Bank": str, "To Bank": str, "Account": str, "Account.1": str}


def is_amlworld(columns) -> bool:
    return {"Is Laundering", "Account.1", "From Bank", "To Bank", "Payment Format"} <= set(columns)


def _ts(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, format="%Y/%m/%d %H:%M", errors="coerce")
    return t if t.notna().mean() > 0.99 else pd.to_datetime(s, errors="coerce")


def _norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lstrip("0").replace("", "0")


def _key(df: pd.DataFrame) -> pd.Series:
    return (df["Timestamp"].astype(str).str.strip() + "|" + _norm_id(df["From Bank"]) + "|" + _norm_id(df["Account"]) + "|" +
            _norm_id(df["To Bank"]) + "|" + _norm_id(df["Account.1"]) + "|" + pd.to_numeric(df["Amount Paid"]).round(2).astype(str))


def parse_patterns(lines) -> pd.DataFrame:
    """Parse laundering attempts. Returns the transactions with `pattern_type` (e.g. FAN_OUT, CYCLE) and `pattern_id`."""
    rows, ptype, pid = [], None, -1
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("BEGIN LAUNDERING ATTEMPT"):
            pid += 1
            name = re.sub(r"^BEGIN LAUNDERING ATTEMPT\s*-?\s*", "", line, flags=re.I).split(":")[0].strip()
            ptype = re.sub(r"[\s\-]+", "_", name.upper()) or "UNKNOWN"
            continue
        if line.upper().startswith("END LAUNDERING ATTEMPT"):
            ptype = None
            continue
        if ptype is not None:
            parts = next(csv.reader([line]))
            if len(parts) >= 10:
                rows.append(parts[:11] + [None] * (11 - len(parts[:11])) + [ptype, pid])
    cols = AML_COLS + ["pattern_type", "pattern_id"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def attach_patterns(df: pd.DataFrame, patterns: pd.DataFrame) -> pd.DataFrame:
    """Add pattern_type / pattern_id to the transactions (post-hoc analysis only; never features)."""
    df = df.copy()
    df["pattern_type"], df["pattern_id"] = "none", -1
    if len(patterns):
        p = patterns.assign(_k=_key(patterns)).drop_duplicates("_k").set_index("_k")
        k = _key(df)
        hit = k.isin(p.index)
        df.loc[hit, "pattern_type"] = k[hit].map(p.pattern_type).values
        df.loc[hit, "pattern_id"] = k[hit].map(p.pattern_id).astype(int).values
    unclassified = (pd.to_numeric(df["Is Laundering"]) == 1) & (df["pattern_type"] == "none")
    df.loc[unclassified, "pattern_type"] = "UNCLASSIFIED"
    return df


def scan_time(chunks):
    """Pass 1 over the Timestamp column only: returns (tmin, rows per hour since tmin)."""
    ts = pd.concat([_ts(c["Timestamp"]) for c in chunks]).dropna()
    tmin = ts.min().floor("h")
    hours = ((ts - tmin) // pd.Timedelta(hours=1)).astype(int).values
    return tmin, np.bincount(hours)


def choose_window(hour_counts: np.ndarray, start_frac: float, rows: int, by: str = "time") -> tuple[int, int]:
    """Hours [h0, h1) holding about `rows` consecutive transactions.
    by="time": start `start_frac` of the way through the CLOCK span (can land in a sparse tail: AMLworld's activity thins out late).
    by="rows": start where `start_frac` of all transactions have already happened (always lands in dense data)."""
    n = len(hour_counts)
    if rows <= 0:
        return 0, n
    if by == "rows":
        h0 = int(np.searchsorted(np.cumsum(hour_counts), start_frac * hour_counts.sum()))
    else:
        h0 = int(start_frac * n)
    h0 = min(h0, n - 1)
    cum = np.cumsum(hour_counts[h0:])
    h1 = h0 + int(np.searchsorted(cum, rows)) + 1
    return h0, min(h1, n)


def adapt(df: pd.DataFrame, tmin: pd.Timestamp) -> pd.DataFrame:
    """Map AMLworld to the pipeline schema. Accounts are keyed by bank + account id; `type` is the payment format."""
    t = _ts(df["Timestamp"])
    paid, recv = pd.to_numeric(df["Amount Paid"]), pd.to_numeric(df["Amount Received"])
    up = lambda s: s.astype(str).str.upper().str.replace(r"[\s\-]+", "_", regex=True)
    out = pd.DataFrame({
        "step": ((t - tmin) // pd.Timedelta(hours=1)).astype("int64") + 1,
        "type": up(df["Payment Format"]),
        "amount": paid,
        "nameOrig": df["From Bank"].astype(str) + "_" + df["Account"].astype(str),
        "oldbalanceOrg": 0.0, "newbalanceOrig": 0.0,
        "nameDest": df["To Bank"].astype(str) + "_" + df["Account.1"].astype(str),
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        "isFraud": pd.to_numeric(df["Is Laundering"]).astype(int),
        "amount_received": recv,
        "receiving_currency": up(df["Receiving Currency"]),
        "payment_currency": up(df["Payment Currency"]),
        "cross_bank": (df["From Bank"].astype(str) != df["To Bank"].astype(str)).astype(int),
        "same_currency": (df["Receiving Currency"].astype(str) == df["Payment Currency"].astype(str)).astype(int),
        "log_amount_ratio": np.log((paid + 1) / (recv + 1)),
    })
    for c in ("pattern_type", "pattern_id"):
        if c in df.columns:
            out[c] = df[c].values
    return out
