"""How identifying is each id-like column? Reads ONLY the columns you name (low RAM).

    python tools/check_id_columns.py data/01_raw/Fraud_Detection_Dataset.csv ip_address card_id device_id --entity customer_id

For each column: share of values that are unique (near 100% = useless for linking), median transactions per value, and
(with --entity) the share of values used by 2+ distinct entities, i.e. real shared infrastructure.
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("columns", nargs="+")
    ap.add_argument("--entity", default=None, help="e.g. customer_id: report sharing across this column")
    a = ap.parse_args()
    cols = a.columns + ([a.entity] if a.entity else [])
    df = pd.read_csv(a.path, usecols=cols)
    n = len(df)
    print(f"rows: {n:,}")
    for c in a.columns:
        vc = df[c].value_counts()
        share = 100 * len(vc) / n
        verdict = "NEAR-UNIQUE: useless for linking" if share > 95 else "some repetition" if share > 50 else "strongly repeated"
        line = f"{c:16s} distinct {len(vc):>10,} ({share:5.1f}% of rows) | median tx per value {vc.median():.0f} | {verdict}"
        if a.entity:
            per = df.groupby(c)[a.entity].nunique()
            line += f" | values shared by 2+ {a.entity}s: {100 * (per > 1).mean():.2f}% (max {int(per.max())})"
        print(line)


if __name__ == "__main__":
    main()
