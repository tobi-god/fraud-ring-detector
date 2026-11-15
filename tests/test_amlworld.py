"""Track A: AMLworld adapter, pattern labels, windowing, causal account-flow features, pattern-level evaluation."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from fraud_ring_detector import amlworld as aw
from fraud_ring_detector.pipelines.fraud_detection.nodes import (_flow_features, build_features, evaluate_models,
                                                                 evaluate_patterns, prepare_dataset, train_baselines,
                                                                 train_hybrid)
from fraud_ring_detector.synthetic import make_synthetic_aml

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_flow_features_are_causal_and_hand_checkable():
    # A->B 100, B->C 80, C->A 50, B->A 10   (accounts A=0, B=1, C=2)
    df = pd.DataFrame({"src": [0, 1, 2, 1], "dst": [1, 2, 0, 0], "amount": [100.0, 80.0, 50.0, 10.0]})
    f = _flow_features(df)
    expm1 = lambda s: np.expm1(f[s].values)
    assert np.allclose(expm1("src_in_amt_before"), [0, 100, 80, 100])      # what the sender had received BEFORE this row
    assert np.allclose(expm1("src_out_amt_before"), [0, 0, 0, 80])         # what the sender had sent BEFORE this row
    assert np.allclose(expm1("dst_in_amt_before"), [0, 0, 0, 50])          # receiver's earlier receipts (not this one)
    assert list(f["reverse_seen"]) == [0.0, 0.0, 0.0, 1.0]                 # B->A reverses the earlier A->B
    df2 = pd.DataFrame({"src": [0, 1], "dst": [1, 0], "amount": [5.0, 5.0]})
    assert list(_flow_features(df2)["reverse_seen"]) == [0.0, 1.0]         # second transfer reverses the first
    assert 0 <= _flow_features(df).filter(like="pass_through").values.max() <= 1


def test_pattern_parser_and_attachment():
    df, text = make_synthetic_aml(n_tx=3000, n_accounts=800, per_pattern=2, seed=1)
    p = aw.parse_patterns(text.splitlines())
    assert {"FAN_OUT", "FAN_IN", "CYCLE", "GATHER_SCATTER", "SCATTER_GATHER", "BIPARTITE", "STACK", "RANDOM"} == set(p.pattern_type)
    m = aw.attach_patterns(df, p)
    assert (m.pattern_type[m["Is Laundering"] == 1] != "none").all()       # every laundering row is typed or UNCLASSIFIED
    assert (m.pattern_type[m["Is Laundering"] == 0] == "none").all()
    assert (m.pattern_type != "none").sum() == int(df["Is Laundering"].sum())


def test_window_selection_covers_requested_rows():
    counts = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
    assert aw.choose_window(counts, 0.5, 25) == (5, 8)
    assert aw.choose_window(counts, 0.0, 0) == (0, 10)
    assert aw.choose_window(counts, 0.95, 1000)[1] == 10                   # never runs past the end


def test_window_by_rows_is_dense_and_time_windows_can_hit_a_sparse_tail(tmp_path):
    counts = np.r_[np.full(80, 1000), np.full(20, 2)]                     # dense for 80 hours, then a thin tail
    assert aw.choose_window(counts, 0.9, 5000, "time")[0] == 90          # clock position lands in the tail
    h0, h1 = aw.choose_window(counts, 0.9, 5000, "rows")
    assert counts[h0:h1].sum() >= 5000 or h1 == len(counts)               # row position stays in dense data
    assert aw.choose_window(counts, 0.5, 5000, "rows")[0] < 80
    df, text = make_synthetic_aml(n_tx=6000, n_accounts=1500, per_pattern=3, seed=2)
    tail = df.tail(3).copy(); tail["Timestamp"] = "2022/09/30 12:00"
    f = tmp_path / "HI-Tail_Trans.csv"; pd.concat([df, tail]).to_csv(f, index=False)
    (tmp_path / "HI-Tail_Patterns.txt").write_text(text)
    run = lambda *extra: subprocess.run([sys.executable, os.path.join(ROOT, "filter_data.py"), "--input", str(f), "--window-rows", "2000", *extra],
                                        capture_output=True, text=True, cwd=tmp_path)
    bad = run("--window-start", "0.99")
    assert bad.returncode != 0 and "thins out" in (bad.stdout + bad.stderr)   # the guard refuses a near-empty window
    assert run("--window-start", "0.5", "--window-by", "rows").returncode == 0


def test_real_file_code_path_with_duplicate_account_columns(tmp_path):
    df, text = make_synthetic_aml(n_tx=6000, n_accounts=1500, per_pattern=3, seed=2)
    f = tmp_path / "HI-Mock_Trans.csv"; df.to_csv(f, index=False)
    (tmp_path / "HI-Mock_Patterns.txt").write_text(text)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "filter_data.py"), "--input", str(f), "--window-rows", "0"],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr + r.stdout
    out = pd.read_csv(tmp_path / "data/01_raw/amlworld_sample.csv")
    assert {"nameOrig", "nameDest", "type", "isFraud", "step", "pattern_type", "pattern_id", "cross_bank"} <= set(out.columns)
    assert int(out.isFraud.sum()) == int(df["Is Laundering"].sum())
    assert out.step.min() >= 1 and "kedro run --env amlworld" in r.stdout


@pytest.fixture(scope="module")
def aml_ready():
    df, text = make_synthetic_aml(n_tx=12_000, n_accounts=3000, per_pattern=8, seed=3)
    raw = aw.attach_patterns(df, aw.parse_patterns(text.splitlines()))
    tmin = aw._ts(raw.Timestamp).min().floor("h")
    ds = prepare_dataset(aw.adapt(raw, tmin), {"seed": 0, "max_rows": 100_000, "split_quantiles": [0.5, 0.7]})
    feats = {"seq_len": 5, "drop_balance_features": True, "account_flow_features": True, "entity_columns": [],
             "exclude_columns": ["pattern_type", "pattern_id", "fraud_type"]}
    f = build_features(ds, feats)
    model = {"seed": 0, "hidden": 16, "gnn_out": 8, "cnn_channels": 8, "dropout": 0.1, "learning_rate": 0.01,
             "weight_decay": 0.0, "epochs": 4, "patience": 3, "focal_alpha": 0.75, "focal_gamma": 2.0}
    base, hyb = train_baselines(ds, f, {"seed": 0}), train_hybrid(ds, f, model)
    return ds, f, base, hyb


def test_pattern_columns_never_become_features(aml_ready):
    ds, f, *_ = aml_ready
    assert f["tx_causal"].shape[1] > 9                                    # account-flow features present
    assert f["n_nodes"] == f["n_core"]                                    # account-to-account: no entity nodes


def test_evaluation_reports_bootstrap_and_pattern_recall(aml_ready):
    ds, f, base, hyb = aml_ready
    ev = {"decision_threshold": None, "sweep_thresholds": [0.5], "negative_sampling_fraction": 1.0,
          "analysis_columns": ["pattern_type"], "bootstrap_samples": 20, "pattern_column": "pattern_type", "min_pattern_tx": 1}
    metrics, _ = evaluate_models(ds, base, hyb, ev)
    assert metrics["bootstrap"]["n_boot"] > 0 and len(metrics["bootstrap"]["delta_M3_minus_M2"]["ci95"]) == 2
    pr = evaluate_patterns(ds, base, hyb, metrics, ev)
    assert {"model", "pattern_type", "tx_recall", "instances_any_flagged"} <= set(pr.columns)
    assert len(pr) > 0 and pr.tx_recall.between(0, 1).all()
    assert (pr.instances_majority_flagged <= pr.instances_any_flagged).all()


def test_budget_methods_handle_tied_scores():
    from fraud_ring_detector.pipelines.fraud_detection.nodes import _flags_at_volume, _threshold_for_budget
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.01).astype(int)
    p = np.where(rng.random(20000) < 0.3, 0.9, rng.random(20000) * 0.8)       # 30% of rows share the SAME top score
    p[y == 1] = np.where(rng.random(int(y.sum())) < 0.5, 0.9, p[y == 1])
    pi = 0.01
    rate = lambda f: 1e4 * (((f & (y == 1)).sum() / y.sum()) * pi + ((f & (y == 0)).sum() / (y == 0).sum()) * (1 - pi))
    for b in (5, 20, 50):
        assert rate(p >= _threshold_for_budget(p, y, pi, b)) <= b + 1e-9      # a threshold can never overshoot the budget
        f = _flags_at_volume(p, y, pi, b)
        assert rate(f) <= b + 1e-9 and rate(f) > 0.5 * b                      # rank-based flags land close to it, even with ties
    assert rate(_flags_at_volume(p, y, pi, 5)) <= rate(_flags_at_volume(p, y, pi, 50))


def test_run_all_summary_reads_a_window_folder(tmp_path):
    import json
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import run_all
    w = tmp_path / "aml_w0.5"; w.mkdir()
    ev = {"20": {"recall": 0.2, "recall_by_type": {"pattern_type": {"UNCLASSIFIED": {"n": 5, "recall": 0.1}}}}, "50": {"recall": 0.3, "recall_by_type": {}}}
    def mod(name, pr):
        return {"model": name, "pr_auc": pr, "at_equal_volume": ev, "recall_by_type": {"pattern_type": {"CYCLE": {"n": 4, "recall": 0.5}}}}
    m = {"baseline_tabular_full": mod("M1 x", 0.05), "baseline_graph_features": mod("M2 x", 0.15), "hybrid_sage_cnn": mod("M3 x", 0.04),
         "bootstrap": {"pr_auc_ci95": {"baseline_tabular_full": [0.03, 0.06], "baseline_graph_features": [0.1, 0.2], "hybrid_sage_cnn": [0.02, 0.06]},
                       "delta_M2_minus_M1": {"mean": 0.1, "ci95": [0.05, 0.15]}, "delta_M3_minus_M2": {"mean": -0.11, "ci95": [-0.17, -0.06]}}}
    json.dump(m, open(w / "metrics.json", "w"))
    json.dump({"rows": 1000, "fraud": 5, "rate": 0.005, "matched": 0.6}, open(w / "facts.json", "w"))
    pd.DataFrame({"model": ["M1 x", "M2 x", "M3 x"], "pattern_type": ["CYCLE"] * 3, "n_tx": [4] * 3, "tx_recall": [.5] * 3, "n_instances": [2] * 3,
                  "alert_budget_per_10k": [20] * 3, "instances_any_flagged": [1] * 3, "instances_majority_flagged": [1] * 3}).to_csv(w / "pattern_recall.csv", index=False)
    text = run_all.summarise(str(tmp_path))
    assert "| time 0.5 |" in text and "entirely below zero: 1 of 1" in text and "UNCLASSIFIED recall" in text


def test_run_all_summary_excludes_degenerate_windows(tmp_path):
    import json
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import run_all
    def window(name, rows, rate):
        d = tmp_path / name; d.mkdir()
        ev = {"20": {"recall": 0.2, "recall_by_type": {}}, "50": {"recall": 0.3, "recall_by_type": {}}}
        mod = lambda pr: {"model": "M x", "pr_auc": pr, "at_equal_volume": ev, "recall_by_type": {"pattern_type": {"CYCLE": {"n": 4, "recall": 0.5}}}}
        m = {"baseline_tabular_full": mod(0.05), "baseline_graph_features": mod(0.15), "hybrid_sage_cnn": mod(0.04),
             "bootstrap": {"pr_auc_ci95": {k: [0.01, 0.2] for k in ("baseline_tabular_full", "baseline_graph_features", "hybrid_sage_cnn")},
                           "delta_M2_minus_M1": {"mean": 0.1, "ci95": [0.05, 0.15]}, "delta_M3_minus_M2": {"mean": -0.1, "ci95": [-0.2, -0.05]}}}
        json.dump(m, open(d / "metrics.json", "w"))
        json.dump({"rows": rows, "fraud": 5, "rate": rate, "matched": 0.6, "req_rows": 300000}, open(d / "facts.json", "w"))
    window("aml_w0.5", 300000, 0.002); window("aml_w0.7", 374, 0.59)
    text = run_all.summarise(str(tmp_path))
    assert "Excluded: window time 0.7" in text and "over 1 window(s)" in text
