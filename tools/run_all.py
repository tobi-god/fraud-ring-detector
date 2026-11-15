"""Run the whole evaluation unattended and bundle everything into ONE file to send back.

    python tools/run_all.py --aml-input data/01_raw/HI-Small_Trans.csv --windows 0.3 0.5 0.7 --track-b

For each AMLworld window: build the sample, run `kedro run --env amlworld`, archive the outputs. Optionally re-run Track B.
Resumable: a window whose results already exist is skipped (use --force to redo). Ctrl+C is safe.
At the end it writes data/08_reporting/final/final_summary.md and data/08_reporting/results_bundle.zip.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join("data", "08_reporting", "final")
KEEP = ["metrics.json", "pattern_recall.csv", "threshold_sweep.csv", "syndicates.csv", "account_roles.csv"]
MODELS = [("baseline_tabular_full", "M1"), ("baseline_graph_features", "M2"), ("hybrid_sage_cnn", "M3")]


def run(cmd, log_path):
    t = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] $ {' '.join(cmd)}", flush=True)
    env = {**os.environ, "KEDRO_DISABLE_TELEMETRY": "1", "PYTHONUTF8": "1"}
    with open(log_path, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, env=env).returncode
    print(f"    finished in {(time.time() - t) / 60:.1f} min (exit code {rc})", flush=True)
    if rc:
        tail = open(log_path, encoding="utf-8", errors="ignore").read().splitlines()[-8:]
        print("    last lines of the log:\n      " + "\n      ".join(tail), flush=True)
    return rc


def copy_outputs(src_dir, dest):
    os.makedirs(dest, exist_ok=True)
    for f in KEEP:
        if os.path.exists(os.path.join(src_dir, f)):
            shutil.copy(os.path.join(src_dir, f), dest)


def sample_facts(path):
    s = pd.read_csv(path, usecols=["step", "isFraud", "pattern_type"])
    fr = s[s.isFraud == 1]
    typed = ~fr.pattern_type.isin(["none", "UNCLASSIFIED"])
    return {"rows": len(s), "fraud": int(len(fr)), "rate": len(fr) / max(len(s), 1), "hours": f"{int(s.step.min())}-{int(s.step.max())}",
            "matched": float(typed.mean()) if len(fr) else 0.0}


def summarise(final_dir) -> str:
    rows, lines = [], []
    for tag in sorted(d for d in os.listdir(final_dir) if d.startswith(("aml_w", "aml_r"))):
        mp = os.path.join(final_dir, tag, "metrics.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        facts = json.load(open(os.path.join(final_dir, tag, "facts.json"))) if os.path.exists(os.path.join(final_dir, tag, "facts.json")) else {}
        pr = pd.read_csv(os.path.join(final_dir, tag, "pattern_recall.csv")) if os.path.exists(os.path.join(final_dir, tag, "pattern_recall.csv")) else pd.DataFrame()
        r = {"window": ("rows " if tag.startswith("aml_r") else "time ") + tag[5:], **facts}
        r["valid"] = not facts or (facts.get("rows", 0) >= 0.5 * facts.get("req_rows", 0) and facts.get("rate", 0) <= 0.05)
        n_test = sum(v["n"] for v in m["baseline_tabular_full"]["recall_by_type"].get("pattern_type", {}).values())
        r["test_pos"] = n_test
        for key, short in MODELS:
            r[f"{short}_pr"] = m[key]["pr_auc"]
            r[f"{short}_ci"] = m["bootstrap"]["pr_auc_ci95"][key] if "bootstrap" in m else None
            ev = m[key].get("at_equal_volume", {})
            for b in ("20", "50"):
                r[f"{short}_rec{b}"] = ev.get(b, {}).get("recall")
            r[f"{short}_unc20"] = ev.get("20", {}).get("recall_by_type", {}).get("pattern_type", {}).get("UNCLASSIFIED", {}).get("recall")
            if len(pr):
                g = pr[(pr.model.str.startswith(short)) & (pr.pattern_type != "UNCLASSIFIED")]
                r[f"{short}_inst"] = (int(g.n_instances.sum()), int(g.instances_any_flagged.sum()), int(g.instances_majority_flagged.sum()))
        d = m.get("bootstrap", {}).get("delta_M3_minus_M2", {})
        r["d32"], r["d32ci"] = d.get("mean"), d.get("ci95")
        d = m.get("bootstrap", {}).get("delta_M2_minus_M1", {})
        r["d21"], r["d21ci"] = d.get("mean"), d.get("ci95")
        rows.append(r)
    excluded = [r for r in rows if not r["valid"]]
    rows = [r for r in rows if r["valid"]]
    f2 = lambda x: "n/a" if x is None else f"{x:.3f}"
    ci = lambda c: "n/a" if not c else f"{c[0]:.3f} to {c[1]:.3f}"
    lines.append("# Track A: all windows in one table\n")
    lines.append("| window start | rows | laundering rows | matched to pattern | test positives | M1 PR-AUC | M2 PR-AUC | M3 PR-AUC | M2-M1 (95%) | M3-M2 (95%) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['window']} | {r.get('rows', 'n/a')} | {r.get('fraud', 'n/a')} ({100 * r.get('rate', 0):.2f}%) | {100 * r.get('matched', 0):.0f}% | {r['test_pos']} | "
                     f"{f2(r['M1_pr'])} ({ci(r['M1_ci'])}) | {f2(r['M2_pr'])} ({ci(r['M2_ci'])}) | {f2(r['M3_pr'])} ({ci(r['M3_ci'])}) | "
                     f"{f2(r['d21'])} ({ci(r['d21ci'])}) | {f2(r['d32'])} ({ci(r['d32ci'])}) |")
    lines.append("\n## Recall at EQUAL alert volume (per 10,000 transactions), test period\n")
    lines.append("| window | volume | M1 | M2 | M3 | UNCLASSIFIED recall at 20 (M1 / M2 / M3) |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        for b in ("20", "50"):
            unc = " / ".join(f2(r.get(f"{s}_unc20")) for _, s in MODELS) if b == "20" else ""
            lines.append(f"| {r['window']} | {b} | {f2(r.get('M1_rec' + b))} | {f2(r.get('M2_rec' + b))} | {f2(r.get('M3_rec' + b))} | {unc} |")
    lines.append("\n## Listed pattern instances in the test period at 20 alerts per 10,000 (instances / any flagged / majority flagged)\n")
    lines.append("| window | M1 | M2 | M3 |")
    lines.append("|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['window']} | " + " | ".join(str(r.get(f"{s}_inst", "n/a")) for _, s in MODELS) + " |")
    for r in excluded:
        lines.append(f"\n**Excluded: window {r['window']}** has only {r.get('rows')} rows with {100 * r.get('rate', 0):.1f}% laundering "
                     "(activity thins out late in the file), so its scores are meaningless.")
    if rows:
        lines.append("\n## Across windows\n")
        for s in ("M1", "M2", "M3"):
            v = [r[f"{s}_pr"] for r in rows]
            lines.append(f"- {s}: mean PR-AUC {sum(v) / len(v):.3f} (range {min(v):.3f} to {max(v):.3f}) over {len(v)} window(s)")
        neg = sum(1 for r in rows if r["d32ci"] and r["d32ci"][1] < 0)
        lines.append(f"- Windows where the M3 minus M2 interval is entirely below zero: {neg} of {len(rows)}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aml-input", default="data/01_raw/HI-Small_Trans.csv")
    ap.add_argument("--windows", type=float, nargs="*", default=[0.3, 0.5, 0.7])
    ap.add_argument("--row-windows", type=float, nargs="*", default=[],
                    help="windows placed by transaction count (always dense), e.g. 0.85 = start after 85%% of all transactions")
    ap.add_argument("--window-rows", type=int, default=300_000)
    ap.add_argument("--track-b", action="store_true", help="also re-run Track B (needs data/01_raw/banking_sample.csv)")
    ap.add_argument("--aml-env", default="amlworld", help="Kedro environment for the AMLworld windows (e.g. amlworld_longtrain)")
    ap.add_argument("--track-b-env", default="mendeley", help="Kedro environment for Track B (e.g. longtrain)")
    ap.add_argument("--final-dir", default=FINAL, help="where results are archived; use a new folder for a new experiment")
    ap.add_argument("--force", action="store_true", help="redo windows that already have results")
    a = ap.parse_args()
    os.chdir(ROOT)
    final = a.final_dir
    os.makedirs(final, exist_ok=True)
    py, failures, t0 = sys.executable, [], time.time()

    jobs = [("w", w) for w in a.windows] + [("r", w) for w in a.row_windows]
    for kind, w in jobs:
        tag = f"aml_{kind}{w}"; out = os.path.join(final, tag)
        if os.path.exists(os.path.join(out, "metrics.json")) and not a.force:
            print(f"[skip] window {w}: results already in {out}"); continue
        os.makedirs(out, exist_ok=True)
        sample = os.path.join("data", "01_raw", f"amlworld_sample_{kind}{w}.csv")
        if not os.path.exists(sample):
            if run([py, "filter_data.py", "--input", a.aml_input, "--window-start", str(w), "--window-rows", str(a.window_rows),
                    "--window-by", "rows" if kind == "r" else "time"],
                   os.path.join(out, "filter_log.txt")):
                failures.append(f"window {w}: building the sample failed (see {out}/filter_log.txt)"); continue
            shutil.copy(os.path.join("data", "01_raw", "amlworld_sample.csv"), sample)
        else:
            print(f"[reuse] window {w}: {sample}")
            shutil.copy(sample, os.path.join("data", "01_raw", "amlworld_sample.csv"))
        json.dump({**sample_facts(sample), "req_rows": a.window_rows}, open(os.path.join(out, "facts.json"), "w"))
        if run([py, "-m", "kedro", "run", "--env", a.aml_env], os.path.join(out, "kedro_log.txt")):
            failures.append(f"window {w}: kedro run failed (see {out}/kedro_log.txt); try a nearby start such as {round(w + 0.05, 2)}"); continue
        copy_outputs(os.path.join("data", "07_model_output", a.aml_env), out)

    if a.track_b:
        out = os.path.join(final, "trackB")
        if os.path.exists(os.path.join(out, "metrics.json")) and not a.force:
            print("[skip] Track B: results already present")
        elif not os.path.exists(os.path.join("data", "01_raw", "banking_sample.csv")):
            failures.append("Track B: data/01_raw/banking_sample.csv is missing (run filter_data.py with --customer-frac 0.3 first)")
        else:
            os.makedirs(out, exist_ok=True)
            if run([py, "-m", "kedro", "run", "--env", a.track_b_env], os.path.join(out, "kedro_log.txt")):
                failures.append(f"Track B: kedro run failed (see {out}/kedro_log.txt)")
            else:
                copy_outputs(os.path.join("data", "07_model_output", a.track_b_env), out)

    summary = summarise(final)
    if failures:
        summary += "\n## Problems\n\n" + "\n".join(f"- {x}" for x in failures) + "\n"
    open(os.path.join(final, "final_summary.md"), "w", encoding="utf-8").write(summary)
    name = "results_bundle.zip" if final == FINAL else f"{os.path.basename(final.rstrip(os.sep))}_bundle.zip"
    bundle = os.path.join("data", "08_reporting", name)
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _, files in os.walk(final):
            for f in files:
                z.write(os.path.join(base, f), os.path.relpath(os.path.join(base, f), os.path.dirname(final)))
    print(f"\nAll done in {(time.time() - t0) / 60:.0f} min. Send me ONE file: {bundle} ({os.path.getsize(bundle) / 1e6:.1f} MB)")
    if failures:
        print("Some steps failed:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
