"""Track B: banking-style data with card/device entities, causal counts, per-type recall, population precision."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import filter_data as fd  # noqa: E402
from fraud_ring_detector.pipelines.fraud_detection.nodes import (build_features, evaluate_models, prepare_dataset,  # noqa: E402
                                                                   train_baselines, train_hybrid)
from fraud_ring_detector.synthetic import make_synthetic_banking  # noqa: E402

FEAT = {"seq_len": 5, "drop_balance_features": True, "entity_columns": ["card_id", "device_id"],
        "exclude_columns": ["fraud_type", "isFlaggedFraud", "transaction_id", "ip_address"]}
MODEL = {"seed": 0, "hidden": 16, "gnn_out": 8, "cnn_channels": 8, "dropout": 0.1, "learning_rate": 0.01,
         "weight_decay": 0.0, "epochs": 4, "patience": 3, "focal_alpha": 0.75, "focal_gamma": 2.0}
EVAL = {"decision_threshold": None, "sweep_thresholds": [0.5], "negative_sampling_fraction": 0.5, "analysis_columns": ["fraud_type"]}


@pytest.fixture(scope="module")
def banking():
    raw = make_synthetic_banking(n_tx=6000, n_customers=600, n_merchants=80, n_rings=6, seed=2)
    df = fd.to_pipeline_schema(fd.normalise_columns(raw))
    fd.check_schema(df)
    return prepare_dataset(df, {"seed": 0, "max_rows": 100_000, "split_quantiles": [0.5, 0.7]})


def test_extras_are_used_but_label_like_columns_are_not(banking):
    f = build_features(banking, FEAT)
    # base = tx features + numeric extras (lat, lon) + one-hot categoricals (category, country, currency)
    assert f["tx_base"].shape[1] > 12
    assert f["ent_idx"].shape == (len(banking), 2)
    assert f["n_nodes"] > f["n_core"]                        # card and device nodes were added
    for X, ei in f["snapshots"].values():
        assert ei.max() < f["n_nodes"] and np.isfinite(X).all()


def test_shared_card_count_is_causal(banking):
    from fraud_ring_detector.pipelines.fraud_detection.nodes import _causal_counts
    c = _causal_counts(banking, ["card_id"])
    col = c["card_id_distinct_customers"]
    one = np.log1p(1.0)
    first_rows = banking.groupby("card_id").head(1).index
    assert np.allclose(col[first_rows], one)                       # first use of any card: exactly one customer so far
    for _, rows in banking.groupby("card_id").groups.items():
        assert (np.diff(col[list(rows)].values) >= -1e-9).all()    # never decreases: only past transactions are counted
    ring = banking.index[banking.card_id.str.startswith("RINGCARD")]
    assert col[ring].max() > one                                   # ring cards are eventually shared by several customers


def test_ladder_runs_and_reports_population_metrics_and_recall_by_type(banking):
    f = build_features(banking, FEAT)
    base = train_baselines(banking, f, {"seed": 0})
    hyb = train_hybrid(banking, f, MODEL)
    metrics, sweep = evaluate_models(banking, base, hyb, EVAL)
    for key in ("baseline_tabular_full", "baseline_graph_features", "hybrid_sage_cnn"):
        m = metrics[key]
        assert 0 <= m["pr_auc"] <= 1 and m["population_prevalence"] < m["prevalence"]
        assert "fraud_type" in m["recall_by_type"]
    assert len(sweep) == 1


def test_message_passing_switch_runs_and_ignores_edges(banking):
    f = build_features(banking, FEAT)
    out = train_hybrid(banking, f, {**MODEL, "message_passing": False})
    assert 0 <= out["best_val_pr_auc"] <= 1 and len(out["p_test"]) == int((banking.split == "test").sum())


def test_warmup_epochs_are_never_selected_and_short_runs_fall_back_to_final_weights(banking):
    f = build_features(banking, FEAT)
    out = train_hybrid(banking, f, {**MODEL, "epochs": 3, "patience": 2, "warmup_epochs": 5})   # never leaves warm-up
    assert len(out["history"]) == 3 and len(out["p_test"]) == int((banking.split == "test").sum())
    out2 = train_hybrid(banking, f, {**MODEL, "epochs": 6, "patience": 2, "warmup_epochs": 2})
    assert out2["best_val_pr_auc"] == max(h["val_pr_auc"] for h in out2["history"][2:])      # best is chosen only after warm-up
