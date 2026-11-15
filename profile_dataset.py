"""Quick triage of any transaction CSV BEFORE you build on it.

    python profile_dataset.py data/01_raw/Fraud_Detection_Dataset.csv

Answers: how many positives per time split? do entities repeat (is there a graph)? do ids share devices/IPs?
does a plain tabular model already solve it? is any single column suspiciously predictive (leakage)?
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

LABELS = ["isFraud", "is_fraud", "fraud", "Is Laundering", "label", "is_laundering"]
TIMES = ["timestamp", "Timestamp", "step", "date", "datetime", "time"]


def pick(cols, options):
    low = {c.lower(): c for c in cols}
    return next((low[o.lower()] for o in options if o.lower() in low), None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--label"); ap.add_argument("--time")
    ap.add_argument("--nrows", type=int, default=400_000, help="rows to read (RAM guard); 0 = the WHOLE file (recommended for time-ordered files)")
    a = ap.parse_args()
    df = pd.read_csv(a.path, nrows=(a.nrows or None))
    y_col = a.label or pick(df.columns, LABELS); t_col = a.time or pick(df.columns, TIMES)
    if not y_col:
        sys.exit(f"Could not find a label column among {list(df.columns)}; pass --label NAME")
    y = df[y_col].astype(int)
    print(f"\n== 1. SIZE AND BASE RATE\nrows read: {len(df):,} | columns: {len(df.columns)} | positives: {int(y.sum()):,} ({100*y.mean():.3f}%)")
    for c in df.columns:
        if "fraud_type" in c.lower() or "typolog" in c.lower() or "pattern" in c.lower():
            print(f"\n   {c} counts:\n" + df[c].fillna("none").value_counts().head(10).to_string())

    if t_col:
        order = df[t_col] if pd.api.types.is_numeric_dtype(df[t_col]) else pd.to_datetime(df[t_col], errors="coerce")
        df = df.assign(_o=order).sort_values("_o").reset_index(drop=True); y = df[y_col].astype(int)
    n = len(df); sp = {"train": slice(0, int(.6*n)), "val": slice(int(.6*n), int(.8*n)), "test": slice(int(.8*n), n)}
    print("\n== 2. POSITIVES PER TIME SPLIT (60/20/20 by time)")
    for k, s in sp.items():
        print(f"   {k:5s}: {int(y.iloc[s].sum()):6,} positives of {s.stop-s.start:,}  ({100*y.iloc[s].mean():.3f}%)")
    if min(int(y.iloc[s].sum()) for s in sp.values()) < 30:
        print("   WARNING: fewer than 30 positives in a split -> metrics will be very noisy")
    shares = [int(y.iloc[s].sum()) / max(int(y.sum()), 1) for s in sp.values()]
    if a.nrows and max(shares) > 0.6:
        print(f"   WARNING: {100*max(shares):.0f}% of positives sit in one split. Time-ordered files put different behaviour in "
              "different periods, so reading only the first rows is unrepresentative. Re-run with --nrows 0.")

    id_cols = [c for c in df.columns if c != y_col and (c.lower().endswith("_id") or any(k in c.lower() for k in ("name", "account", "ip_address")))
               and df[c].nunique() < 0.95 * n]
    print("\n== 3. DO ENTITIES REPEAT? (a graph needs repeated counterparties)")
    for c in id_cols:
        vc = df[c].value_counts(); rep = (vc >= 2).mean()
        f_rep = y[df[c].map(vc) >= 2].mean(); f_one = y[df[c].map(vc) < 2].mean() if (df[c].map(vc) < 2).any() else float("nan")
        print(f"   {c:18s} unique {df[c].nunique():>9,} | ids seen 2+ times: {100*rep:5.1f}% | median tx/id {vc.median():.0f} | fraud rate: repeat {100*f_rep:.2f}% vs single {100*f_one:.2f}%")

    cust = next((c for c in id_cols if any(k in c.lower() for k in ("customer", "orig", "account", "sender", "user"))), None)
    print("\n== 4. SHARED INFRASTRUCTURE (rings often share devices / IPs / cards)")
    if cust:
        for c in id_cols:
            if c == cust or any(k in c.lower() for k in ("merchant", "dest")): continue
            shared = df.groupby(c)[cust].transform("nunique") >= 2
            if shared.any():
                print(f"   {c:18s} shared by 2+ {cust}s on {100*shared.mean():5.1f}% of tx | fraud rate shared {100*y[shared].mean():.2f}% vs not {100*y[~shared].mean():.2f}% "
                      f"| {100*(shared & (y == 1)).sum()/max(int(y.sum()),1):.1f}% of all fraud tx")
            else:
                print(f"   {c:18s} never shared across {cust}s")
    else:
        print("   (no customer-like column detected)")

    print("\n== 5. IS IT ALREADY EASY FOR A TABULAR MODEL? (and any leakage?)")
    Xn = df.drop(columns=[y_col, "_o"], errors="ignore")
    textual = [c for c in Xn.columns if not pd.api.types.is_numeric_dtype(Xn[c]) and not pd.api.types.is_bool_dtype(Xn[c])]
    idlike = [c for c in Xn.columns if "id" in c.lower() and Xn[c].nunique() > 0.95 * len(Xn)]
    drop = [c for c in textual if Xn[c].nunique() > 30] + [c for c in id_cols if c in Xn] + idlike
    X = pd.get_dummies(Xn.drop(columns=list(set(drop)), errors="ignore"), dummy_na=False).astype(float).fillna(0)
    X = X.loc[:, X.nunique() > 1]
    print(f"   using {X.shape[1]} features (ids, free text and timestamps excluded): " + ", ".join(list(X.columns)[:25]) + (" ..." if X.shape[1] > 25 else ""))
    ytr, yte = y.iloc[sp["train"]], y.iloc[sp["test"]]
    if ytr.sum() > 5 and yte.sum() > 5:
        gb = HistGradientBoostingClassifier(max_iter=150, class_weight="balanced", random_state=0).fit(X.iloc[sp["train"]], ytr)
        p = gb.predict_proba(X.iloc[sp["test"]])[:, 1]
        ap_, base = average_precision_score(yte, p), yte.mean()
        print(f"   plain GBM on non-id columns: test PR-AUC {ap_:.3f} (random would be {base:.4f}), ROC-AUC {roc_auc_score(yte, p):.3f}")
        print("   -> " + ("already very strong: a graph model has little room to add value" if ap_ > 0.9 else
                           "moderate: graph signals could add value" if ap_ > 0.3 else "hard for tabular: promising for graph signals"))
        sus = []
        for c in X.columns:
            v = X[c].iloc[sp["train"]]
            if v.nunique() > 1:
                try:
                    auc = roc_auc_score(ytr, v); auc = max(auc, 1 - auc)
                    if auc > 0.98: sus.append((c, auc))
                except ValueError: pass
        for c, auc in sus: print(f"   LEAKAGE SUSPECT: '{c}' alone gives ROC-AUC {auc:.3f}")
    else:
        print("   too few positives in train/test to fit a baseline")
    print("\nDone. Use sections 2-4 to decide whether this dataset can support a graph claim.")


if __name__ == "__main__":
    main()
