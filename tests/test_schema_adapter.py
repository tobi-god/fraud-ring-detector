"""filter_data.py must map banking-style schemas (customer/merchant/timestamp) onto the pipeline schema."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import filter_data as fd  # noqa: E402


def test_banking_schema_is_adapted_and_fraud_type_is_not_a_feature_column():
    raw = pd.DataFrame({"transaction_id": [1, 2], "timestamp": ["2025-01-01 00:00:00", "2025-01-01 05:30:00"],
                        "customer_id": ["C1", "C2"], "merchant_id": ["M1", "M1"], "transaction_type": ["purchase", "transfer"],
                        "amount": [10.0, 20.0], "is_fraud": [0, 1], "fraud_type": ["none", "ring"]})
    df = fd.to_pipeline_schema(fd.normalise_columns(raw))
    fd.check_schema(df)
    assert {"nameOrig", "nameDest", "type", "isFraud", "step"} <= set(df.columns)
    assert list(df.step) == [1, 6] and set(df.type) == {"PURCHASE", "TRANSFER"}
    assert "fraud_type" in fd.LEAKY_LABEL_COLUMNS


def test_paysim_snake_case_headers_are_normalised():
    cols = fd.normalise_columns(pd.DataFrame(columns=["is_fraud", "name_orig", "old_balance_org"])).columns
    assert list(cols) == ["isFraud", "nameOrig", "oldbalanceOrg"]


def test_customer_sampling_keeps_all_fraud_customers_and_whole_histories(tmp_path, monkeypatch):
    import subprocess
    from fraud_ring_detector.synthetic import make_synthetic_banking
    raw = make_synthetic_banking(n_tx=5000, n_customers=500, n_merchants=60, n_rings=4, seed=3)
    f = tmp_path / "raw.csv"; raw.to_csv(f, index=False)
    root = os.path.join(os.path.dirname(__file__), "..")
    monkeypatch.chdir(tmp_path)
    r = subprocess.run([sys.executable, os.path.join(root, "filter_data.py"), "--input", str(f), "--customer-frac", "0.3"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = pd.read_csv(tmp_path / "data/01_raw/banking_sample.csv")
    fraud_customers = set(raw.loc[raw.is_fraud == 1, "customer_id"])
    assert fraud_customers <= set(out.nameOrig)                                         # every fraud customer kept
    full_hist = raw.groupby("customer_id").size()
    kept_hist = out.groupby("nameOrig").size()
    assert (kept_hist == full_hist[kept_hist.index]).all()                              # histories are complete
    assert "negative_sampling_fraction" in r.stdout and out.nameOrig.nunique() < raw.customer_id.nunique()
