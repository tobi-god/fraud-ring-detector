"""Run the pipeline for several seeds and summarise mean +/- std per model.

    python tools/run_seeds.py --seeds 1 2 3 4 5 [--env mendeley]

Each run varies BOTH data.seed (negative sampling) and model.seed (initialisation). Per-run metrics are kept in
data/08_reporting/<env>/runs/seed_<n>/ and the summary in data/08_reporting/<env>/seed_summary.md (+ .csv).
"""
import argparse, json, os, shutil, subprocess, sys

import pandas as pd

METRICS = ["pr_auc", "roc_auc", "precision", "recall", "f1", "population_precision", "alerts_per_10k"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--env", default=None); a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(root)
    rows = []
    src_dir = os.path.join("data", "07_model_output", a.env) if a.env else os.path.join("data", "07_model_output")
    rep_dir = os.path.join("data", "08_reporting", a.env or "default")
    for s in a.seeds:
        cmd = [sys.executable, "-m", "kedro", "run", "--params", f"data.seed={s},model.seed={s}"] + (["--env", a.env] if a.env else [])
        print(f"\n=== seed {s}: {' '.join(cmd[2:])}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            sys.exit(f"kedro run failed for seed {s}")
        dest = os.path.join(rep_dir, "runs", f"seed_{s}"); os.makedirs(dest, exist_ok=True)
        for f in ("metrics.json", "threshold_sweep.csv", "syndicates.csv", "pattern_recall.csv"):
            if os.path.exists(os.path.join(src_dir, f)):
                shutil.copy(os.path.join(src_dir, f), dest)
        m = json.load(open(os.path.join(dest, "metrics.json")))
        for key, v in m.items():
            if isinstance(v, dict) and "pr_auc" in v:
                rows.append({"seed": s, "model": v["model"], **{k: v[k] for k in METRICS}})
    df = pd.DataFrame(rows)
    agg = df.groupby("model")[METRICS].agg(["mean", "std"]).round(4)
    agg.columns = [f"{a_}_{b}" for a_, b in agg.columns]
    out = rep_dir; os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "seed_runs.csv"), index=False); agg.to_csv(os.path.join(out, "seed_summary.csv"))
    lines = ["| model | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for model, r in agg.iterrows():
        lines.append(f"| {model} | " + " | ".join(f"{r[f'{k}_mean']:.3f} ± {0 if pd.isna(r[f'{k}_std']) else r[f'{k}_std']:.3f}" for k in METRICS) + " |")
    open(os.path.join(out, "seed_summary.md"), "w").write("\n".join(lines) + "\n")
    print(f"\nSeeds {a.seeds}: mean ± std\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
