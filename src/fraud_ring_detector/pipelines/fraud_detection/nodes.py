"""Pure node functions for the fraud-ring pipeline (no file I/O; the Data Catalog handles persistence)."""
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

from .model import HybridGraphCNN, focal_loss
from .topology import assign_roles, build_flow_graph, directed_communities, find_cycles, overlapping_memberships

logger = logging.getLogger(__name__)
TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
# Columns derived from balances. In PaySim, fraudulent transfers are cancelled, so balances can leak the label.
BALANCE_FEATURES = ["orig_delta", "dest_delta", "err_orig", "err_dest", "drain_ratio", "orig_zeroed"]
REQUIRED = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
            "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"]


# ----------------------------------------------------------------------------- 1. data
def prepare_dataset(raw: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Stratified cap (keep all fraud), chronological sort, TIME-based split, shared node index."""
    missing = set(REQUIRED) - set(raw.columns)
    if missing:
        raise ValueError(f"raw_transactions is missing columns: {sorted(missing)}")
    fraud, legit = raw[raw.isFraud == 1], raw[raw.isFraud == 0]
    n_keep = max(min(len(legit), params["max_rows"] - len(fraud)), 0)
    df = pd.concat([fraud, legit.sample(n=n_keep, random_state=params["seed"])])
    df = df.sort_values("step", kind="stable").reset_index(drop=True)

    q1, q2 = df.step.quantile(params["split_quantiles"]).values
    df["split"] = np.where(df.step <= q1, "train", np.where(df.step <= q2, "val", "test"))
    for s in ("train", "val", "test"):
        n = int(df.loc[df.split == s, "isFraud"].sum())
        if n < 5:
            raise ValueError(f"Only {n} fraud rows in '{s}' split. Use more data: a larger --window-rows or a different --window-start "
                             "(AMLworld), a higher --customer-frac / --legit-frac, or a larger max_rows.")

    # ONE node space for senders and receivers (fixes the old bipartite index mismatch in SAGEConv)
    nodes = pd.Index(pd.unique(np.concatenate([df.nameOrig.values, df.nameDest.values])))
    df["src"] = nodes.get_indexer(df.nameOrig)
    df["dst"] = nodes.get_indexer(df.nameDest)
    logger.info("dataset: %d tx, %d accounts, fraud rate %.3f%%", len(df), len(nodes), 100 * df.isFraud.mean())
    return df


# ----------------------------------------------------------------------------- 2. features
CORE_COLUMNS = set(REQUIRED) | {"isFlaggedFraud", "split", "src", "dst"}


def _tx_features(d: pd.DataFrame, drop_balance: bool = False, types=TYPES) -> pd.DataFrame:
    sl = lambda s: np.sign(s) * np.log1p(s.abs())
    f = pd.DataFrame(index=d.index)
    f["log_amt"] = np.log1p(d.amount)
    f["orig_delta"] = sl(d.oldbalanceOrg - d.newbalanceOrig)
    f["dest_delta"] = sl(d.newbalanceDest - d.oldbalanceDest)
    f["err_orig"] = sl(d.oldbalanceOrg - d.amount - d.newbalanceOrig)
    f["err_dest"] = sl(d.oldbalanceDest + d.amount - d.newbalanceDest)
    f["drain_ratio"] = (d.amount / (d.oldbalanceOrg + 1)).clip(0, 2)
    f["orig_zeroed"] = ((d.newbalanceOrig == 0) & (d.oldbalanceOrg > 0)).astype(float)
    f["hour"] = (d.step % 24) / 23.0
    for t in types:
        f["t_" + t] = (d.type == t).astype(float)
    return f.drop(columns=BALANCE_FEATURES) if drop_balance else f


def _extra_columns(df: pd.DataFrame, params: dict, entity_cols: list) -> tuple[list, list]:
    """Remaining columns usable as tabular features: low-cardinality categoricals and numeric columns.
    Label-like and id-like columns are excluded (params.exclude_columns, entity columns, near-unique ids)."""
    spec = params.get("extra_columns", "auto")
    if spec in (None, False, []):
        return [], []
    excl = CORE_COLUMNS | set(params.get("exclude_columns", [])) | set(entity_cols) | {"nameOrig", "nameDest"}
    pool = [c for c in df.columns if c not in excl] if spec == "auto" else [c for c in spec if c in df.columns]
    cats, nums = [], []
    for c in pool:
        s, nu = df[c], df[c].nunique()
        if nu < 2:
            continue
        if pd.api.types.is_bool_dtype(s):
            cats.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            if not ("id" in c.lower() and nu > 0.95 * len(df)):
                nums.append(c)
        elif nu <= params.get("max_cardinality", 30):
            cats.append(c)
    return cats, nums


def _flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal account-to-account flow signals (use only transactions BEFORE each row):
    what the sender received/sent before, what the receiver sent before, the pass-through ratio (money in vs out: mules),
    and whether the reverse transfer already happened (round-tripping)."""
    n = len(df)
    idx = np.arange(n)
    amt = df.amount.values.astype(float)
    ev = pd.DataFrame({"acct": np.r_[df.src.values, df.dst.values], "idx": np.r_[idx, idx],
                       "is_dst": np.r_[np.zeros(n), np.ones(n)], "amt": np.r_[amt, amt]})
    ev = ev.sort_values(["acct", "idx", "is_dst"], kind="stable")      # sender event precedes receiver event of the same row
    g = ev.groupby("acct")
    ev["in_cnt"] = g["is_dst"].cumsum()
    ev["out_cnt"] = (1 - ev.is_dst).groupby(ev.acct).cumsum()
    ev["in_amt"] = (ev.amt * ev.is_dst).groupby(ev.acct).cumsum()
    ev["out_amt"] = (ev.amt * (1 - ev.is_dst)).groupby(ev.acct).cumsum()
    s_ev = ev[ev.is_dst == 0].sort_values("idx")
    d_ev = ev[ev.is_dst == 1].sort_values("idx")
    src_in, src_out = s_ev.in_amt.values, s_ev.out_amt.values - amt          # sender: received / sent BEFORE this row
    dst_out, dst_in = d_ev.out_amt.values, d_ev.in_amt.values - amt          # receiver: sent / received BEFORE this row
    ratio = lambda a, b: np.where(np.maximum(a, b) > 0, np.minimum(a, b) / np.maximum(np.maximum(a, b), 1e-12), 0.0)
    N = int(max(df.src.max(), df.dst.max())) + 1
    fwd = df.src.values.astype(np.int64) * N + df.dst.values
    rev = df.dst.values.astype(np.int64) * N + df.src.values
    first = pd.Series(idx).groupby(fwd).min()
    seen = first.reindex(rev).values
    out = {"src_in_cnt_before": np.log1p(s_ev.in_cnt.values), "dst_out_cnt_before": np.log1p(d_ev.out_cnt.values),
           "src_in_amt_before": np.log1p(src_in), "src_out_amt_before": np.log1p(src_out),
           "dst_in_amt_before": np.log1p(dst_in), "dst_out_amt_before": np.log1p(dst_out),
           "src_pass_through": ratio(src_in, src_out), "dst_pass_through": ratio(dst_in, dst_out),
           "reverse_seen": (seen < idx).astype(float)}
    return pd.DataFrame(out).astype(float)


def _causal_counts(df: pd.DataFrame, ent_cols: list) -> pd.DataFrame:
    """Counts that only use information available up to each transaction (rows are time-ordered)."""
    out = {"cust_tx_so_far": df.groupby("src").cumcount(), "merchant_tx_so_far": df.groupby("dst").cumcount()}
    new_pair = ~df.duplicated(["src", "dst"])
    out["new_counterparty"] = new_pair.astype(float)
    out["cust_distinct_counterparties"] = new_pair.groupby(df.src).cumsum()
    out["merchant_distinct_senders"] = new_pair.groupby(df.dst).cumsum()
    for e in ent_cols:
        first = ~df.duplicated([e, "src"])
        out[f"{e}_tx_so_far"] = df.groupby(e).cumcount()
        out[f"{e}_distinct_customers"] = first.groupby(df[e]).cumsum()
        out[f"cust_distinct_{e}"] = first.groupby(df.src).cumsum()
        out[f"{e}_shared"] = (out[f"{e}_distinct_customers"] > 1).astype(float)
    c = pd.DataFrame(out)
    for col in c.columns:
        if not (col.endswith("_shared") or col == "new_counterparty"):
            c[col] = np.log1p(c[col].astype(float))
    return c.astype(float)


def _snapshot(df, n_core, n_nodes, ent_idx, max_step, drop_bal, bidirectional):
    """Graph visible at `max_step`: account/merchant stats + entity (card/device) nodes and edges. No later data."""
    m = df.step.values <= max_step
    d = df[m].assign(orig_zero=lambda x: ((x.newbalanceOrig == 0) & (x.oldbalanceOrg > 0)).astype(float))
    out_g = d.groupby("src").agg(out_deg=("amount", "size"), out_sum=("amount", "sum"), out_mean=("amount", "mean"),
                                 out_uniq=("dst", "nunique"), out_drain=("orig_zero", "mean"))
    in_g = d.groupby("dst").agg(in_deg=("amount", "size"), in_sum=("amount", "sum"), in_mean=("amount", "mean"),
                                in_uniq=("src", "nunique"))
    core = pd.DataFrame(index=np.arange(n_core)).join(out_g).join(in_g).fillna(0.0)
    for c in ["out_deg", "out_sum", "out_mean", "out_uniq", "in_deg", "in_sum", "in_mean", "in_uniq"]:
        core[c] = np.log1p(core[c])
    if drop_bal:
        core = core.drop(columns=["out_drain"])
    n_ent, c0 = ent_idx.shape[1], core.shape[1]
    X = np.zeros((n_nodes, c0 + 3 + 1 + n_ent), dtype=np.float32)       # core stats | entity stats | node-type one-hot
    X[:n_core, :c0] = core.values
    X[:n_core, c0 + 3] = 1.0
    src, edges = d.src.values, [np.stack([d.src.values, d.dst.values])]
    for j in range(n_ent):
        nodes = ent_idx[m, j]
        g = pd.DataFrame({"n": nodes, "c": src, "a": d.amount.values}).groupby("n").agg(
            t=("a", "size"), u=("c", "nunique"), s=("a", "sum"))
        X[g.index.values, c0] = np.log1p(g.t.values)
        X[g.index.values, c0 + 1] = np.log1p(g.u.values)
        X[g.index.values, c0 + 2] = np.log1p(g.s.values)
        X[g.index.values, c0 + 4 + j] = 1.0
        edges.append(np.stack([src, nodes]))
    ei = np.concatenate(edges, 1).astype(np.int64)
    if bidirectional:                       # SAGEConv passes messages source->target; without reverse edges senders get none
        ei = np.concatenate([ei, ei[::-1]], 1)
    _, keep = np.unique(ei[0] * n_nodes + ei[1], return_index=True)
    return X, ei[:, np.sort(keep)]


def build_features(df: pd.DataFrame, params: dict) -> dict[str, Any]:
    """Tabular features, causal shared-entity counts, per-split graph snapshots (with card/device nodes) and the
    last-K sequence tensor. Every statistic is computed from data available at the time it is used."""
    K, E = params["seq_len"], len(df)
    drop_bal = bool(params.get("drop_balance_features", False))
    ent_cols = [c for c in (params.get("entity_columns") or []) if c in df.columns]
    missing = set(params.get("entity_columns") or []) - set(ent_cols)
    if missing:
        logger.warning("entity columns not found and ignored: %s", sorted(missing))
    df = df.assign(**{e: df[e].fillna("__missing__").astype(str) for e in ent_cols})
    types = sorted(df.type.astype(str).unique())
    txf = _tx_features(df, drop_bal, types)
    cats, nums = _extra_columns(df, params, ent_cols)
    parts = [txf] + ([df[nums].astype(float)] if nums else []) + \
            ([pd.get_dummies(df[cats].astype(str), dtype=float)] if cats else [])
    base = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
    tr = (df.split == "train").values
    zs = lambda M: ((M - M[tr].mean()) / M[tr].std().replace(0, 1).fillna(1)).fillna(0).values.astype(np.float32)
    TXB = zs(base)
    causal = _causal_counts(df, ent_cols)
    if params.get("account_flow_features", False):
        causal = pd.concat([causal, _flow_features(df)], axis=1)
    CAUS = zs(causal)
    logger.info("features: %d base (%d numeric extras, %d categorical extras -> one-hot) + %d causal | entities: %s | "
                "balance features %s", TXB.shape[1], len(nums), len(cats), CAUS.shape[1], ent_cols or "none",
                "dropped" if drop_bal else "kept")
    logger.info("extras used: numeric=%s categorical=%s", nums, cats)

    n_core = int(max(df.src.max(), df.dst.max())) + 1
    ent_idx, off = np.zeros((E, len(ent_cols)), dtype=np.int64), n_core
    for j, e in enumerate(ent_cols):
        codes, uniq = pd.factorize(df[e])
        ent_idx[:, j] = codes + off
        off += len(uniq)
    n_nodes = off
    bidir = bool(params.get("bidirectional_edges", True))
    split_end = {s: int(df.loc[df.split == s, "step"].max()) for s in ("train", "val", "test")}
    snaps = {s: _snapshot(df, n_core, n_nodes, ent_idx, ms, drop_bal, bidir) for s, ms in split_end.items()}
    nmu, nsd = snaps["train"][0].mean(0), snaps["train"][0].std(0) + 1e-6
    snaps = {s: ((X - nmu) / nsd, ei) for s, (X, ei) in snaps.items()}

    order = np.lexsort((np.arange(E), df.step.values, df.src.values))      # by account, then time
    src_s, steps_s = df.src.values[order], df.step.values[order]
    cols = list(txf.columns)
    pick = [cols.index(c) for c in ["log_amt", "orig_delta", "err_orig", "drain_ratio", "orig_zeroed"] if c in cols] + \
           [cols.index("t_" + t) for t in types]
    base_seq = TXB[order][:, pick]
    gap = np.zeros(E, dtype=np.float32)
    same_prev = np.r_[False, src_s[1:] == src_s[:-1]]
    gap[same_prev] = np.log1p(steps_s[1:][same_prev[1:]] - steps_s[:-1][same_prev[1:]])
    base_seq = np.concatenate([base_seq, gap[:, None]], 1)
    seq_sorted = np.zeros((E, base_seq.shape[1], K), dtype=np.float32)
    pos = np.arange(E)
    for k in range(1, K + 1):
        ok = (pos - k >= 0) & (src_s[np.clip(pos - k, 0, None)] == src_s)
        seq_sorted[ok, :, k - 1] = base_seq[pos[ok] - k]
    SEQ = np.zeros_like(seq_sorted)
    SEQ[order] = seq_sorted

    # hand-built "graph features" for the tabular baseline: snapshot stats of sender and receiver + history summary
    seq_sum = np.concatenate([SEQ.mean(2), (np.abs(SEQ).sum(1) > 0).mean(1, keepdims=True)], 1)
    TXG = np.zeros((E, 2 * snaps["train"][0].shape[1] + seq_sum.shape[1]), dtype=np.float32)
    for s in ("train", "val", "test"):
        rows = np.where(df.split.values == s)[0]
        X = snaps[s][0]
        TXG[rows] = np.concatenate([X[df.src.values[rows]], X[df.dst.values[rows]], seq_sum[rows]], 1)
    return {"tx_base": TXB, "tx_causal": CAUS, "tx_graph": TXG, "seq": SEQ, "snapshots": snaps,
            "n_nodes": n_nodes, "n_core": n_core, "ent_idx": ent_idx}


# ----------------------------------------------------------------------------- 3. models
def _best_f1_threshold(y, p) -> float:
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    return float(th[np.nanargmax(f1[:-1])]) if len(th) else 0.5


def train_baselines(df: pd.DataFrame, features: dict, params: dict) -> dict[str, Any]:
    """Baseline ladder. M1: tabular only. M2: M1 + causal shared-entity counts + hand-built graph/sequence features.
    The hybrid model must beat M2 (not just M1) before graph learning can be credited with anything."""
    y = df.isFraud.values
    i = {s: np.where(df.split.values == s)[0] for s in ("train", "val", "test")}
    sets = {"tabular_full": features["tx_base"],
            "graph_features": np.concatenate([features["tx_base"], features["tx_causal"], features["tx_graph"]], 1)}
    out = {}
    for name, X in sets.items():
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced",
                                            random_state=params["seed"])
        gb.fit(X[i["train"]], y[i["train"]])
        out[name] = {"p_val": gb.predict_proba(X[i["val"]])[:, 1], "p_test": gb.predict_proba(X[i["test"]])[:, 1],
                     "n_features": int(X.shape[1])}
    return out


def train_hybrid(df: pd.DataFrame, features: dict, params: dict) -> dict[str, Any]:
    """Full-batch training with focal loss; early stopping on validation PR-AUC; scores val/test transactions."""
    seed = params.get("seed", 42)
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = lambda a: torch.as_tensor(a).to(dev)
    y = df.isFraud.values.astype(np.float32)
    idx = {s: t(np.where(df.split.values == s)[0]) for s in ("train", "val", "test")}
    TX = t(np.concatenate([features["tx_base"], features["tx_causal"]], 1))
    SEQ, Y, SRC, DST, ENT = t(features["seq"]), t(y), t(df.src.values), t(df.dst.values), t(features["ent_idx"])
    SN = {s: (t(X), t(ei)) for s, (X, ei) in features["snapshots"].items()}

    model = HybridGraphCNN(SN["train"][0].shape[1], TX.shape[1], SEQ.shape[1], ENT.shape[1], params["hidden"],
                           params["gnn_out"], params["cnn_channels"], params["dropout"],
                           params.get("message_passing", True)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])

    def run(split):
        i = idx[split]; x, ei = SN[split]
        return model(x, ei, SRC[i], DST[i], ENT[i], TX[i], SEQ[i])

    warm = int(params.get("warmup_epochs", 0))
    best, bad, state, hist = -1.0, 0, None, []
    y_val = y[idx["val"].cpu().numpy()]
    for ep in range(1, params["epochs"] + 1):
        model.train(); opt.zero_grad()
        loss = focal_loss(run("train"), Y[idx["train"]], params["focal_alpha"], params["focal_gamma"])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            ap = average_precision_score(y_val, torch.sigmoid(run("val")).cpu().numpy())
        hist.append({"epoch": ep, "loss": float(loss.item()), "val_pr_auc": float(ap)})
        if ep > warm:                       # warm-up epochs are never selected and never count toward patience: with few
            if ap > best:                   # validation positives, epoch 1 can "win" by chance
                best, bad, state = ap, 0, {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        if ep % 10 == 0 or ep == 1:
            logger.info("epoch %3d | loss %.4f | val PR-AUC %.4f", ep, float(loss.item()), ap)
        if bad >= params["patience"]:
            logger.info("early stop at epoch %d", ep); break
    if state is None:                       # never past warm-up: use the final weights
        state = {k: v.clone() for k, v in model.state_dict().items()}; best = float(hist[-1]["val_pr_auc"])
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        p_val, p_test = torch.sigmoid(run("val")).cpu().numpy(), torch.sigmoid(run("test")).cpu().numpy()
    return {"p_val": p_val, "p_test": p_test, "best_val_pr_auc": float(best), "history": hist}


# ----------------------------------------------------------------------------- 4. evaluation
def _report(name, p, y, thr, pi, df_test, analysis_cols):
    pred = (p >= thr).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    rec = tp / max(int(y.sum()), 1)
    fpr = fp / max(int((y == 0).sum()), 1)
    denom = rec * pi + fpr * (1 - pi)                       # alert rate in the full (un-sampled) population
    rep = {"model": name, "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
           "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0)), "threshold": float(thr), "prevalence": float(y.mean()),
           "population_prevalence": float(pi), "population_precision": float(rec * pi / denom) if denom > 0 else 0.0,
           "alerts_per_10k": float(10_000 * denom)}
    by_type = {}
    for col in analysis_cols:
        if col in df_test.columns:
            col_res = {}
            for v in sorted(df_test.loc[y == 1, col].dropna().astype(str).unique()):
                mask = (y == 1) & (df_test[col].astype(str).values == v)
                col_res[v] = {"n": int(mask.sum()), "recall": float(pred[mask].mean()) if mask.any() else None}
            by_type[col] = col_res
    rep["recall_by_type"] = by_type
    return rep


def _cluster_bootstrap(y, probs: dict, groups, n_boot: int, seed: int) -> dict:
    """Resample SENDER accounts (a customer's transactions are correlated) and recompute PR-AUC with weights."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    out = {k: [] for k in probs}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv]
        if (w * y).sum() == 0:
            continue
        for k, p in probs.items():
            out[k].append(average_precision_score(y, p, sample_weight=w))
    return {k: np.array(v) for k, v in out.items()}


def _threshold_for_budget(p, y, pi, budget: float) -> float:
    """Validation-calibrated threshold: the lowest score whose COMPLETE tie group keeps the population-adjusted alert rate
    within `budget` alerts per 10,000. If even the top tie group exceeds the budget, no alerts are allowed.
    Applied unchanged to the test period, so the realised rate can differ (score drift): it is reported alongside."""
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    P, N = max(int(y.sum()), 1), max(int((y == 0).sum()), 1)
    rate = 1e4 * ((tp / P) * pi + (fp / N) * (1 - pi))
    ends = np.r_[ps[1:] != ps[:-1], True]                      # only cut between distinct scores
    ok = np.where(ends & (rate <= budget))[0]
    return float(ps[ok[-1]]) if len(ok) else float(ps[0]) + 1e-9


def _flags_at_volume(p, y, pi, budget: float, seed: int = 0):
    """Equal-volume comparison on the test period: flag the top-ranked transactions up to the point where the
    population-adjusted alert rate reaches `budget` per 10,000. Ties are broken at random, so scores that saturate at the
    same value cannot make a model look better or worse than it is."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(p)), -p))
    ys = y[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    P, N = max(int(y.sum()), 1), max(int((y == 0).sum()), 1)
    rate = 1e4 * ((tp / P) * pi + (fp / N) * (1 - pi))
    k = int(np.searchsorted(rate, budget, side="right"))
    flags = np.zeros(len(p), dtype=bool)
    flags[order[:k]] = True
    return flags


def _volume_report(flags, y, pi, df_test, analysis_cols):
    tp, fp = int((flags & (y == 1)).sum()), int((flags & (y == 0)).sum())
    rec, fpr = tp / max(int(y.sum()), 1), fp / max(int((y == 0).sum()), 1)
    denom = rec * pi + fpr * (1 - pi)
    out = {"recall": float(rec), "population_precision": float(rec * pi / denom) if denom > 0 else 0.0,
           "realised_alerts_per_10k": float(1e4 * denom), "alerts_in_test": int(flags.sum()), "recall_by_type": {}}
    for col in analysis_cols:
        if col in df_test.columns:
            res = {}
            for v in sorted(df_test.loc[y == 1, col].dropna().astype(str).unique()):
                mask = (y == 1) & (df_test[col].astype(str).values == v)
                res[v] = {"n": int(mask.sum()), "recall": float(flags[mask].mean()) if mask.any() else None}
            out["recall_by_type"][col] = res
    return out


def evaluate_models(df: pd.DataFrame, baselines: dict, hybrid: dict, params: dict):
    """Held-out (future) test metrics for the whole ladder, population-adjusted precision, per-fraud-type recall."""
    y_val = df.isFraud.values[df.split.values == "val"]
    te = df[df.split == "test"]
    y_test = te.isFraud.values
    f = params.get("negative_sampling_fraction", 1.0)
    pos = int(df.isFraud.sum()); neg = len(df) - pos
    pi = pos / (pos + neg / f)
    ladder = {"baseline_tabular_full": ("M1 GradBoost, tabular only", baselines["tabular_full"]),
              "baseline_graph_features": ("M2 GradBoost + graph/sequence features", baselines["graph_features"]),
              "hybrid_sage_cnn": ("M3 Hybrid GraphSAGE+CNN", hybrid)}
    fixed = params["decision_threshold"]
    metrics = {}
    for key, (label, m) in ladder.items():
        thr = fixed if fixed is not None else _best_f1_threshold(y_val, m["p_val"])
        metrics[key] = _report(label, m["p_test"], y_test, thr, pi, te, params.get("analysis_columns", []))
        r = metrics[key]
        logger.info("%s | PR-AUC %.4f ROC-AUC %.4f P %.3f R %.3f | population precision %.3f, %.1f alerts/10k",
                    label, r["pr_auc"], r["roc_auc"], r["precision"], r["recall"], r["population_precision"], r["alerts_per_10k"])
    nb = int(params.get("bootstrap_samples", 0))
    if nb > 0 and te.src.nunique() > 5:
        bs = _cluster_bootstrap(y_test, {k: v[1]["p_test"] for k, v in ladder.items()}, te.src.values, nb, 0)
        ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
        d21 = bs["baseline_graph_features"] - bs["baseline_tabular_full"]
        d32 = bs["hybrid_sage_cnn"] - bs["baseline_graph_features"]
        metrics["bootstrap"] = {"unit": "sender accounts", "n_boot": int(len(d32)),
                                "pr_auc_ci95": {k: ci(v) for k, v in bs.items()},
                                "delta_M2_minus_M1": {"mean": float(d21.mean()), "ci95": ci(d21), "share_positive": float((d21 > 0).mean())},
                                "delta_M3_minus_M2": {"mean": float(d32.mean()), "ci95": ci(d32), "share_positive": float((d32 > 0).mean())}}
        logger.info("bootstrap 95%% CI of PR-AUC delta: M2-M1 %s | M3-M2 %s", ci(d21), ci(d32))
    for key, (label, m) in ladder.items():                 # equal operating points: same alert budget for every model
        at = {}
        for b in params.get("alert_budgets_per_10k", []) or []:
            thr_b = _threshold_for_budget(m["p_val"], y_val, pi, b)
            r = _report(label, m["p_test"], y_test, thr_b, pi, te, params.get("analysis_columns", []))
            at[str(b)] = {"threshold": thr_b, "recall": r["recall"], "population_precision": r["population_precision"],
                          "realised_alerts_per_10k": r["alerts_per_10k"], "recall_by_type": r["recall_by_type"]}
        vol = {str(b): _volume_report(_flags_at_volume(m["p_test"], y_test, pi, b), y_test, pi, te,
                                       params.get("analysis_columns", []))
               for b in (params.get("alert_budgets_per_10k", []) or [])}
        if vol:
            metrics[key]["at_equal_volume"] = vol            # threshold-free, tie-safe: the fair comparison
            logger.info("%s | EQUAL VOLUME recall at %s alerts/10k: %s", label, list(vol), [round(v["recall"], 3) for v in vol.values()])
        if at:
            metrics[key]["at_alert_budget"] = at             # validation-calibrated: what an operator would actually deploy
            logger.info("%s | at budgets %s alerts/10k: recall %s", label, list(at),
                        [round(v["recall"], 3) for v in at.values()])
    metrics["note"] = (f"Time-held-out test set. negative_sampling_fraction={f} -> population prevalence {pi:.5f}. "
                       "'f1' is plain F1. Compare M3 with M2, not only M1.")
    rows = []
    for t in params["sweep_thresholds"]:
        pred = (hybrid["p_test"] >= t).astype(int)
        rows.append({"threshold": t, "alerts": int(pred.sum()),
                     "true_fraud_caught": int(((pred == 1) & (y_test == 1)).sum()), "total_fraud": int(y_test.sum()),
                     "precision": round(float(precision_score(y_test, pred, zero_division=0)), 3),
                     "recall": round(float(recall_score(y_test, pred, zero_division=0)), 3)})
    return metrics, pd.DataFrame(rows)


def evaluate_patterns(df: pd.DataFrame, baselines: dict, hybrid: dict, metrics: dict, params: dict) -> pd.DataFrame:
    """Pattern-level (ring-level) recall on the test period, per model and laundering pattern type.
    A pattern INSTANCE counts as detected-any if at least one of its transactions is flagged, and as
    detected-majority if at least half are. Pattern labels are never used as features."""
    col = params.get("pattern_column") or "pattern_type"
    cols = ["model", "pattern_type", "n_tx", "tx_recall", "n_instances", "alert_budget_per_10k", "instances_any_flagged", "instances_majority_flagged"]
    if col not in df.columns or "pattern_id" not in df.columns:
        logger.info("no pattern labels in the data -> pattern evaluation skipped")
        return pd.DataFrame(columns=cols)
    te = df[df.split == "test"].reset_index(drop=True)
    pos = te[te.isFraud == 1]
    if pos.empty:
        return pd.DataFrame(columns=cols)
    ladder = {"baseline_tabular_full": baselines["tabular_full"], "baseline_graph_features": baselines["graph_features"],
              "hybrid_sage_cnn": hybrid}
    rows = []
    budget = params.get("pattern_alert_budget")
    for key, m in ladder.items():
        if budget:
            flag = _flags_at_volume(m["p_test"], te.isFraud.values, metrics[key]["population_prevalence"], budget).astype(int)
        else:
            flag = (m["p_test"] >= metrics[key]["threshold"]).astype(int)
        fp = pos.assign(flag=flag[pos.index.values])
        for ptype, g in fp.groupby(col):
            inst = g[g.pattern_id >= 0].groupby("pattern_id").flag.agg(["size", "mean"])
            inst = inst[inst["size"] >= params.get("min_pattern_tx", 2)]
            rows.append({"model": metrics[key]["model"], "pattern_type": ptype, "n_tx": int(len(g)),
                         "tx_recall": float(g.flag.mean()), "n_instances": int(len(inst)),
                         "alert_budget_per_10k": budget if budget else None,
                         "instances_any_flagged": int((inst["mean"] > 0).sum()),
                         "instances_majority_flagged": int((inst["mean"] >= 0.5).sum())})
    out = pd.DataFrame(rows, columns=cols).round(4)
    logger.info("pattern evaluation: %d pattern types x %d models", out.pattern_type.nunique(), len(ladder))
    return out


# ----------------------------------------------------------------------------- 5. topology
def _flagged_test_frame(df, hybrid, metrics, key="hybrid_sage_cnn"):
    te = df[df.split == "test"].copy()
    te["score"] = hybrid["p_test"]
    te["flag"] = (te.score >= metrics[key]["threshold"]).astype(int)
    return te


def detect_syndicates(df: pd.DataFrame, baselines: dict, hybrid: dict, metrics: dict, params: dict):
    """Directed, amount-aware syndicate analysis on the FLAGGED subgraph.

    Returns (syndicate table, account table with roles + overlap, interactive HTML)."""
    from pyvis.network import Network
    t = params["topology"]
    source = t.get("score_source", "hybrid")                  # "hybrid" (M3) or "graph_features" (M2, usually the better model)
    scores, key = (hybrid, "hybrid_sage_cnn") if source == "hybrid" else (baselines["graph_features"], "baseline_graph_features")
    te = _flagged_test_frame(df, scores, metrics, key)
    logger.info("syndicate analysis uses the flags of %s", metrics[key]["model"])
    # Topology uses a looser "candidate" cut-off than the alert threshold: a ring is only visible as a cycle if
    # EVERY hop is in the graph, and any per-edge model misses some hops. Precision per syndicate will reflect this.
    cut = t["candidate_threshold_factor"] * metrics[key]["threshold"]
    fl = te[te.score >= cut]
    names = pd.concat([df[["src", "nameOrig"]].set_axis(["i", "n"], axis=1),
                       df[["dst", "nameDest"]].set_axis(["i", "n"], axis=1)]).drop_duplicates("i").set_index("i").n
    G = build_flow_graph(fl)
    deg = pd.concat([df.src, df.dst]).value_counts()
    logger.info("candidate tx %d (score >= %.3f) | accounts with degree 1 overall: %.1f%% (PaySim has few repeated counterparties)",
                len(fl), cut, 100 * (deg == 1).mean())

    comm_of = directed_communities(G, params["seed"], t["louvain_resolution"])
    memberships = overlapping_memberships(G, params["seed"], t["max_line_graph_edges"]) if t["overlap"] else {}
    cycles = find_cycles(G, t["max_cycle_len"], t["max_cycles"])
    in_cycle = {n for c in cycles for n in c}

    roles = assign_roles(G, t["fan_threshold"], t["role_near_zero"], t["role_pass_through"])
    roles["syndicate"] = roles.account_idx.map(comm_of).fillna(-1).astype(int)
    roles["in_cycle"] = roles.account_idx.isin(in_cycle)
    roles["n_overlap_memberships"] = roles.account_idx.map(lambda n: len(memberships.get(n, ())))
    roles["overlaps_multiple"] = roles.n_overlap_memberships > 1
    roles["account"] = roles.account_idx.map(names)
    truth = pd.concat([fl[["src", "isFraud"]].set_axis(["i", "f"], axis=1),
                       fl[["dst", "isFraud"]].set_axis(["i", "f"], axis=1)]).groupby("i").f.max()
    roles["touches_true_fraud"] = roles.account_idx.map(truth).fillna(0).astype(int)

    cyc_by_comm = {}
    for c in cycles:
        for cid in {comm_of[n] for n in c}:
            cyc_by_comm[cid] = cyc_by_comm.get(cid, 0) + 1
    fl = fl.assign(syndicate=[comm_of.get(int(s), -1) for s in fl.src])
    summary = (fl.groupby("syndicate").agg(n_tx=("score", "size"), n_accounts=("src", "nunique"),
                                           mean_score=("score", "mean"), true_fraud=("isFraud", "sum"),
                                           total_amount=("amount", "sum"))
                 .assign(precision=lambda d: d.true_fraud / d.n_tx).reset_index())
    rc = roles.groupby("syndicate").agg(originators=("role", lambda r: int((r == "originator").sum())),
                                        mules=("role", lambda r: int((r == "mule").sum())),
                                        collectors=("role", lambda r: int((r == "collector").sum())),
                                        fan_hubs=("fan_out_hub", "sum"), overlap_accounts=("overlaps_multiple", "sum"))
    summary = summary.merge(rc, left_on="syndicate", right_index=True, how="left")
    summary["cycles_within_or_touching"] = summary.syndicate.map(cyc_by_comm).fillna(0).astype(int)
    summary = summary.sort_values("total_amount", ascending=False).reset_index(drop=True)
    logger.info("syndicates %d | cycles found %d | accounts in >1 overlapping community %d",
                len(summary), len(cycles), int(roles.overlaps_multiple.sum()))

    shape = {"originator": "triangle", "collector": "square", "mule": "diamond", "mixed": "dot"}
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", directed=True,
                  cdn_resources="in_line")
    for r in roles.itertuples():
        net.add_node(int(r.account_idx), label=str(r.account)[-6:], group=int(r.syndicate), shape=shape[r.role],
                     borderWidth=4 if r.in_cycle else 1,
                     title=f"{r.account} | role {r.role} | syndicate {r.syndicate} | in-cycle {r.in_cycle} | "
                           f"communities {r.n_overlap_memberships}")
    for s, d, e in G.edges(data=True):
        net.add_edge(int(s), int(d), value=float(e["score"]), title=f"amount {e['amount']:,.0f} | tx {e['n_tx']} "
                                                                    f"| max score {e['score']:.2f}")
    return summary.round(4), roles.drop(columns=["account_idx"]).round(4), net.generate_html()


# ----------------------------------------------------------------------------- 6. optional Neo4j
def sync_to_neo4j(df: pd.DataFrame, hybrid: dict, metrics: dict, params: dict) -> dict:
    """Optional. Secrets come from env vars, never from files. Writes flagged tx with CREATE in batches."""
    if not params["enabled"]:
        return {"status": "skipped", "reason": "neo4j.enabled is false in parameters.yml"}
    uri, user, pwd = (os.environ.get(k) for k in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"))
    if not (uri and user and pwd):
        return {"status": "skipped", "reason": "NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD not set"}
    database = params["database"] or os.environ.get("NEO4J_DATABASE") or user
    from neo4j import GraphDatabase
    fl = _flagged_test_frame(df, hybrid, metrics).query("flag == 1")
    rows = [dict(o=o, d=d, step=int(s), amount=float(a), type=t, score=float(sc), fraud=int(f))
            for o, d, s, a, t, sc, f in zip(fl.nameOrig, fl.nameDest, fl.step, fl.amount, fl.type, fl.score, fl.isFraud)]
    q = """UNWIND $rows AS r
           MERGE (a:Account {id: r.o}) MERGE (b:Account {id: r.d})
           CREATE (a)-[:TRANSACTED {step: r.step, amount: r.amount, type: r.type, score: r.score, fraud: r.fraud}]->(b)"""
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    try:
        driver.verify_connectivity()
        driver.execute_query("CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE",
                             database_=database)
        for i in range(0, len(rows), params["batch_size"]):
            driver.execute_query(q, rows=rows[i:i + params["batch_size"]], database_=database)
    finally:
        driver.close()
    return {"status": "ok", "database": database, "transactions_written": len(rows)}
