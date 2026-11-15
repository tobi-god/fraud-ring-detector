"""Why are some laundering transactions 'UNCLASSIFIED'? Run after filter_data.py on AMLworld.

    python tools/check_pattern_match.py data/01_raw/HI-Small_Patterns.txt [data/01_raw/amlworld_sample.csv]

For every UNCLASSIFIED laundering row it asks: does the same sender->receiver pair appear in the patterns file (then the
row format or amount/time differs: a MATCHING problem), do only the accounts appear (a partial overlap), or is it absent
altogether (the simulator labelled it laundering but did not list it in a pattern: a genuinely unlisted row)?
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from fraud_ring_detector import amlworld as aw  # noqa: E402


def norm(bank, acct):
    return bank.astype(str).str.strip().str.lstrip("0").replace("", "0") + "_" + acct.astype(str).str.strip().str.lstrip("0").replace("", "0")


def main():
    pat_path = sys.argv[1]
    sample = sys.argv[2] if len(sys.argv) > 2 else "data/01_raw/amlworld_sample.csv"
    p = aw.parse_patterns(open(pat_path, encoding="utf-8", errors="ignore").read().splitlines())
    s = pd.read_csv(sample, dtype={"nameOrig": str, "nameDest": str})
    u = s[(s.isFraud == 1) & (s.pattern_type == "UNCLASSIFIED")].copy()
    t = s[(s.isFraud == 1) & (s.pattern_type != "UNCLASSIFIED")]
    print(f"patterns file: {len(p):,} lines, {p.pattern_id.nunique()} attempts | sample: {int(s.isFraud.sum())} laundering rows, "
          f"{len(t)} typed, {len(u)} UNCLASSIFIED")
    if u.empty:
        print("nothing to diagnose"); return
    fix = lambda col: col.astype(str).str.replace(r"^0+(?=.)", "", regex=True)
    sender = norm(p["From Bank"], p["Account"]); receiver = norm(p["To Bank"], p["Account.1"])
    pairs = set(zip(sender, receiver)); accts = set(sender) | set(receiver)
    o = u.nameOrig.str.split("_", n=1, expand=True); d = u.nameDest.str.split("_", n=1, expand=True)
    uo = norm(o[0], o[1]); ud = norm(d[0], d[1])
    pair_hit = pd.Series([(a, b) in pairs for a, b in zip(uo, ud)], index=u.index)
    acct_hit = pd.Series([(a in accts) or (b in accts) for a, b in zip(uo, ud)], index=u.index)
    both = pair_hit.sum(); one = (acct_hit & ~pair_hit).sum(); none = (~acct_hit).sum()
    print(f"\nUNCLASSIFIED rows: {len(u)}")
    print(f"  same sender->receiver pair IS in the patterns file : {both:4d}  <- matching problem (amount/time/format differ)")
    print(f"  only one of the accounts appears in the file       : {one:4d}  <- partial overlap (pattern cut by the window?)")
    print(f"  neither account appears in the patterns file       : {none:4d}  <- unlisted laundering (labelled by the simulator)")
    cols = ["step", "nameOrig", "nameDest", "amount", "type"]
    for label, mask in [("pair in file", pair_hit), ("one account in file", acct_hit & ~pair_hit), ("absent", ~acct_hit)]:
        if mask.any():
            print(f"\nexamples ({label}):\n" + u.loc[mask, cols].head(4).to_string(index=False))
    print("\nReading: mostly 'absent' = the dataset itself leaves a share of laundering unlisted (report pattern results for the "
          "listed share only). Mostly 'pair in file' = a matching bug; send me two example lines from the patterns file.")


if __name__ == "__main__":
    main()
