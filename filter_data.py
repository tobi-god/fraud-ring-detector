"""Create data/01_raw/paysim_sample.csv.

Real data (recommended):  python filter_data.py --input "data/01_raw/Fraud Detection Dataset.csv"
Smoke test (no download): python filter_data.py --synthetic

Sampling is STRATIFIED: every fraud row is kept, legit rows are subsampled. (The old version took the first
50,000 rows, which contain almost no fraud and only the earliest time steps.)
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from fraud_ring_detector import amlworld as aw  # noqa: E402
OUT_DIR = "data/01_raw"
OUT_NAMES = {"paysim": "paysim_sample.csv", "banking": "banking_sample.csv", "amlworld": "amlworld_sample.csv"}
RUN_HINT = {"paysim": "kedro run", "banking": "kedro run --env mendeley", "amlworld": "kedro run --env amlworld"}
CANON = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest",
         "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]
# match on lower-case letters/digits only, so is_fraud, IsFraud and isFraud all map to isFraud
ALIASES = {"oldbalanceorig": "oldbalanceOrg", "isfraudulent": "isFraud", "fraud": "isFraud",
           # banking-style schemas (e.g. the Mendeley multi-pattern dataset): customer -> merchant transactions
           "customerid": "nameOrig", "merchantid": "nameDest", "transactiontype": "type"}
BALANCE = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
LEAKY_LABEL_COLUMNS = ["fraud_type"]   # encodes the label itself: kept only for post-hoc analysis, never a feature


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {"".join(ch for ch in c.lower() if ch.isalnum()): c for c in CANON}
    lookup.update(ALIASES)
    rename = {}
    for col in df.columns:
        key = "".join(ch for ch in str(col).lower() if ch.isalnum())
        if key in lookup and lookup[key] != col:
            rename[col] = lookup[key]
    return df.rename(columns=rename)


def to_pipeline_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Derive `step` (hours since the first transaction) from a timestamp and add neutral balance columns if the
    dataset has none. Without balances you MUST set features.drop_balance_features: true."""
    if "step" not in df.columns:
        tcol = next((c for c in df.columns if c.lower() in ("timestamp", "datetime", "date", "time")), None)
        if tcol is None:
            sys.exit("ERROR: no `step` or timestamp column to order transactions in time.")
        ts = pd.to_datetime(df[tcol], errors="coerce")
        df = df.assign(step=((ts - ts.min()).dt.total_seconds() // 3600).astype("int64") + 1)
    if not all(c in df.columns for c in BALANCE):
        for c in BALANCE:
            df[c] = 0.0
        print("NOTE: dataset has no balance columns -> filled with 0. Set `features.drop_balance_features: true` "
              "in conf/base/parameters.yml.")
    df["type"] = df["type"].astype(str).str.upper().str.replace(" ", "_")
    return df


def check_schema(df: pd.DataFrame) -> None:
    need = [c for c in CANON if c not in ("isFlaggedFraud", "step") and c not in BALANCE]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit("ERROR: this file does not have the PaySim schema.\n"
                 f"  missing columns : {missing}\n  columns found   : {list(df.columns)}\n"
                 "It is probably a different dataset. Send the column list to whoever maintains the adapter, "
                 "or map the columns yourself before running the pipeline.")


def detect_schema(cols) -> str:
    if aw.is_amlworld(cols):
        return "amlworld"
    low = {"".join(ch for ch in str(c).lower() if ch.isalnum()) for c in cols}
    return "banking" if "customerid" in low else "paysim"


def run_amlworld(a, chunk_factory, pattern_lines):
    """IBM AMLworld: choose a contiguous time window (keeps natural prevalence and full graph density inside it),
    attach pattern labels (post-hoc analysis only), map to the pipeline schema."""
    tmin, counts = aw.scan_time(chunk_factory(["Timestamp"]))
    h0, h1 = aw.choose_window(counts, a.window_start, a.window_rows, a.window_by)
    print(f"Time span: {tmin} + {len(counts)} hours | window: hours {h0} to {h1} "
          f"({tmin + pd.Timedelta(hours=h0)} -> {tmin + pd.Timedelta(hours=h1)}) = {int(counts[h0:h1].sum()):,} transactions")
    n_win = int(counts[h0:h1].sum())
    if a.window_rows > 0 and n_win < 0.5 * a.window_rows and not a.allow_small_window:
        sys.exit(f"ERROR: this window holds only {n_win:,} transactions (asked for about {a.window_rows:,}): activity in the file thins out "
                 "here. Use --window-by rows, or an earlier --window-start, or --allow-small-window if you really want it.")
    parts = []
    for chunk in chunk_factory(None):
        h = (aw._ts(chunk["Timestamp"]) - tmin) // pd.Timedelta(hours=1)
        keep = (h >= h0) & (h < h1)
        if keep.any():
            parts.append(chunk[keep])
    raw = pd.concat(parts)
    if pattern_lines is not None:
        raw = aw.attach_patterns(raw, aw.parse_patterns(pattern_lines))
        laund = pd.to_numeric(raw["Is Laundering"]).astype(int) == 1
        typed = ~raw["pattern_type"].isin(["none", "UNCLASSIFIED"])
        share = float(typed[laund].mean()) if laund.any() else 0.0
        print(f"{100 * share:.0f}% of the {int(laund.sum()):,} laundering transactions in the window matched a pattern line.")
        if laund.any() and share < 0.5:
            print("WARNING: most laundering rows did not match the patterns file (different file, or a different row format). "
                  "Pattern-level evaluation will be unreliable; check the first lines of the patterns file.")
        pt = raw[raw["pattern_type"] != "none"]
        print(f"Pattern labels attached: {int((pt.pattern_id.unique() >= 0).sum()) if len(pt) else 0} "
              f"pattern instances present in the window; transactions by type: {pt.pattern_type.value_counts().to_dict()}")
    else:
        print("WARNING: no patterns file found. Ring/pattern-level evaluation will be unavailable (pass --patterns PATH).")
    return aw.adapt(raw, tmin)


def main():
    ap = argparse.ArgumentParser(description="Create data/01_raw/<schema>_sample.csv for the pipeline.")
    ap.add_argument("--input", default="data/01_raw/Fraud Detection Dataset.csv")
    ap.add_argument("--schema", choices=["paysim", "banking", "amlworld"], default=None, help="default: detected from the header")
    ap.add_argument("--output", default=None, help="default: data/01_raw/<schema>_sample.csv")
    ap.add_argument("--legit-frac", type=float, default=None,
                    help="fraction of non-fraud rows to keep (default 0.03 for files, 1.0 for mock data). "
                         "Row sampling thins every account's history, which hurts graph and sequence features: "
                         "use 1.0 if the file fits in memory (roughly up to 2M rows).")
    ap.add_argument("--customer-frac", type=float, default=None,
                    help="RAM-friendly sampling that keeps whole histories: keep ALL transactions of every customer who has "
                         "fraud plus this fraction of the other customers (e.g. 0.3). Ignores --legit-frac.")
    ap.add_argument("--window-start", type=float, default=0.5, help="AMLworld: where the time window starts, as a fraction of the time span")
    ap.add_argument("--window-rows", type=int, default=300_000, help="AMLworld: transactions in the window (0 = the whole file; needs lots of RAM)")
    ap.add_argument("--window-by", choices=["time", "rows"], default="time",
                    help="AMLworld: --window-start is a fraction of the clock span (time) or of all transactions (rows; always dense)")
    ap.add_argument("--allow-small-window", action="store_true", help="AMLworld: accept a window with far fewer rows than requested")
    ap.add_argument("--patterns", default=None, help="AMLworld: *_Patterns.txt (default: next to the input, same prefix)")
    ap.add_argument("--synthetic", action="store_true", help="PaySim-style synthetic smoke-test data")
    ap.add_argument("--synthetic-banking", action="store_true", help="banking-style mock data (cards, devices, rings)")
    ap.add_argument("--synthetic-aml", action="store_true", help="AMLworld-style mock data with laundering patterns")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    pattern_lines = None

    if a.synthetic:
        from fraud_ring_detector.synthetic import make_synthetic_paysim
        schema, df, frac = "paysim", make_synthetic_paysim(seed=a.seed), 1.0
        print("Generated SYNTHETIC PaySim-style data (pipeline smoke test only).")
    else:
        if a.synthetic_banking or a.synthetic_aml:
            if a.synthetic_aml:
                from fraud_ring_detector.synthetic import make_synthetic_aml
                mock, text = make_synthetic_aml(seed=a.seed)
                schema, pattern_lines = "amlworld", text.splitlines()
                chunk_factory = lambda cols: [mock if cols is None else mock[cols]]
            else:
                from fraud_ring_detector.synthetic import make_synthetic_banking
                schema, mock = "banking", make_synthetic_banking(seed=a.seed)
                chunk_factory = lambda cols: [mock]
            frac = 1.0 if a.legit_frac is None else a.legit_frac
            print(f"Generated SYNTHETIC {schema} mock data (code-path test only; values are invented).")
        else:
            if not os.path.exists(a.input):
                sys.exit(f"ERROR: '{a.input}' not found. Download a dataset into data/01_raw/ or use a --synthetic* flag.")
            schema = a.schema or detect_schema(pd.read_csv(a.input, nrows=0).columns)
            frac = 0.03 if a.legit_frac is None else a.legit_frac
            dtypes = {c: str for c in aw.STR_COLS}
            chunk_factory = lambda cols: pd.read_csv(a.input, usecols=cols, dtype={k: v for k, v in dtypes.items() if cols is None or k in cols},
                                                     chunksize=500_000) if schema == "amlworld" else pd.read_csv(a.input, chunksize=500_000)
            if schema == "amlworld":
                p = a.patterns or re.sub(r"_Trans\.csv$", "_Patterns.txt", a.input)
                pattern_lines = open(p, encoding="utf-8", errors="ignore").read().splitlines() if os.path.exists(p) else None
        print(f"Schema: {schema}")
        if schema == "amlworld":
            df, frac = run_amlworld(a, chunk_factory, pattern_lines), 1.0
        else:
            make_chunks = lambda: chunk_factory(None)
            keep_customers, neg_total, neg_kept = None, 0, 0
            if a.customer_frac is not None:                       # pass 1: who has fraud, who exists
                fraud_c, all_c = set(), set()
                for chunk in make_chunks():
                    chunk = normalise_columns(chunk)
                    check_schema(chunk)
                    chunk["isFraud"] = chunk["isFraud"].astype(int)
                    fraud_c.update(chunk.loc[chunk.isFraud == 1, "nameOrig"]); all_c.update(chunk["nameOrig"])
                    neg_total += int((chunk.isFraud == 0).sum())
                others = sorted(all_c - fraud_c)
                pick = np.random.default_rng(a.seed).choice(len(others), size=int(a.customer_frac * len(others)), replace=False)
                keep_customers = fraud_c | {others[i] for i in pick}
                print(f"Customers: {len(all_c):,} total | {len(fraud_c):,} with fraud (all kept) | "
                      f"{len(pick):,} of {len(others):,} others kept")
            parts = []
            for chunk in make_chunks():                           # (pass 2) chunked: safe on 8GB RAM
                chunk = normalise_columns(chunk)
                check_schema(chunk)
                chunk["isFraud"] = chunk["isFraud"].astype(int)   # also accepts True/False
                if keep_customers is not None:
                    kept = chunk[chunk.nameOrig.isin(keep_customers)]
                    neg_kept += int((kept.isFraud == 0).sum()); parts.append(kept)
                else:
                    parts.append(pd.concat([chunk[chunk.isFraud == 1],
                                            chunk[chunk.isFraud == 0].sample(frac=frac, random_state=a.seed)]))
            if keep_customers is not None:
                frac = neg_kept / max(neg_total, 1)               # exact share of non-fraud rows retained
            df = to_pipeline_schema(pd.concat(parts))
            tcol = [c for c in df.columns if c.lower() in ("timestamp", "datetime", "date", "time")]
            df = df.drop(columns=tcol)                            # `step` already carries the time order
    first = [c for c in CANON + LEAKY_LABEL_COLUMNS if c in df.columns]
    df = df[first + [c for c in df.columns if c not in first]]   # keep ALL other columns (category, device, card...)
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    out = a.output or os.path.join(OUT_DIR, OUT_NAMES[schema])
    df.to_csv(out, index=False)
    print(f"Saved {out} | shape={df.shape} | fraud rows={int(df.isFraud.sum())} ({100 * df.isFraud.mean():.3f}%)")
    print(f"Columns kept: {list(df.columns)}")
    if frac < 1.0:
        print(f"IMPORTANT: only {frac:.4f} of non-fraud rows were kept. Set `evaluation.negative_sampling_fraction: {frac:.4f}` in "
              "conf/base/parameters.yml (or the environment's parameters.yml) so precision and alerts/10k reflect the real population.")
    print(f"Next: {RUN_HINT[schema]}")


if __name__ == "__main__":
    main()
