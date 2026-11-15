"""Smoke tests: chronological split, shapes, no out-of-range node ids, balance switch."""
import numpy as np

from fraud_ring_detector.pipelines.fraud_detection.nodes import build_features, prepare_dataset
from fraud_ring_detector.synthetic import make_synthetic_paysim

DATA = {"seed": 0, "max_rows": 50_000, "split_quantiles": [0.6, 0.8]}


def _df():
    return prepare_dataset(make_synthetic_paysim(n_legit=8000, n_accounts=3000, n_rings=12, seed=1), DATA)


def test_time_split_is_chronological_and_keeps_fraud():
    raw = make_synthetic_paysim(n_legit=8000, n_accounts=3000, n_rings=12, seed=1)
    df = prepare_dataset(raw, DATA)
    assert df[df.split == "train"].step.max() <= df[df.split == "val"].step.min()
    assert df[df.split == "val"].step.max() <= df[df.split == "test"].step.min()
    assert df.isFraud.sum() == raw.isFraud.sum()          # every fraud row retained


def test_features_shapes_and_shared_node_index():
    df = _df()
    f = build_features(df, {"seq_len": 10})
    assert f["tx_base"].shape[0] == f["tx_causal"].shape[0] == f["tx_graph"].shape[0] == f["seq"].shape[0] == len(df)
    assert f["seq"].shape[2] == 10
    for X, ei in f["snapshots"].values():
        assert X.shape[0] == f["n_nodes"] and ei.max() < f["n_nodes"]   # no out-of-range node ids
    assert all(np.isfinite(f[k]).all() for k in ("tx_base", "tx_causal", "tx_graph"))


def test_bidirectional_edges_let_senders_receive_messages():
    df = _df()
    both = build_features(df, {"seq_len": 10, "bidirectional_edges": True})["snapshots"]["train"][1]
    one = build_features(df, {"seq_len": 10, "bidirectional_edges": False})["snapshots"]["train"][1]
    senders = set(df[df.split == "train"].src)
    assert len(senders & set(both[1].tolist())) > len(senders & set(one[1].tolist()))


def test_drop_balance_features_removes_leaky_columns():
    df = _df()
    full = build_features(df, {"seq_len": 10, "drop_balance_features": False})
    lean = build_features(df, {"seq_len": 10, "drop_balance_features": True})
    assert lean["tx_base"].shape[1] == full["tx_base"].shape[1] - 6
    assert lean["seq"].shape[1] < full["seq"].shape[1]
    assert lean["snapshots"]["train"][0].shape[1] == full["snapshots"]["train"][0].shape[1] - 1
