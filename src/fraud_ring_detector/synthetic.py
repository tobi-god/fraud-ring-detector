"""PaySim-style synthetic data with injected laundering rings. SMOKE-TEST ONLY: fraud here is easy by construction."""
import numpy as np
import pandas as pd


def make_synthetic_paysim(n_legit=60_000, n_accounts=20_000, n_rings=12, ring_size=6, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    accts = np.array([f"C{i:09d}" for i in range(n_accounts)])
    merch = np.array([f"M{i:07d}" for i in range(3000)])
    rows = []
    for _ in range(n_legit):
        step = int(rng.integers(1, 700))
        t = rng.choice(["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"], p=[.5, .2, .15, .1, .05])
        o = rng.choice(accts)
        d = rng.choice(merch) if t == "PAYMENT" else rng.choice(accts)
        amt = float(rng.lognormal(8.0, 1.0)); ob = float(rng.lognormal(10, 1.2))
        rows.append((step, t, amt, o, ob, max(ob - amt, 0) if t != "CASH_IN" else ob + amt, d,
                     float(rng.lognormal(10, 1.2)), 0.0, 0))
    for _ in range(n_rings):
        members = rng.choice(accts, ring_size, replace=False)
        start = int(rng.integers(50, 650))
        for rep in range(3):
            for i in range(ring_size):
                o, d = members[i], members[(i + 1) % ring_size]
                amt = float(rng.lognormal(11.5, .4)); ob = amt * float(rng.uniform(1.0, 1.05))
                rows.append((start + rep * 4 + i, "TRANSFER" if i % 2 == 0 else "CASH_OUT", amt, o, ob, ob - amt,
                             d, 0.0, amt if i % 2 == 0 else 0.0, 1))
    df = pd.DataFrame(rows, columns=["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
                                     "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"])
    df["isFlaggedFraud"] = 0
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def make_synthetic_banking(n_tx=40_000, n_customers=3000, n_merchants=400, n_rings=8, ring_size=4, seed=0) -> pd.DataFrame:
    """Banking-style mock (customer/card/device/merchant, several fraud types, rings that SHARE cards and devices).
    Column names mimic the Mendeley multi-pattern file. Values are invented: code-path tests only."""
    rng = np.random.default_rng(seed)
    cust = np.array([f"CUST{i:05d}" for i in range(n_customers)])
    mer = np.array([f"M{i:04d}" for i in range(n_merchants)])
    m_cat = rng.choice(["food", "fuel", "retail", "travel", "online"], n_merchants)
    m_cty = rng.choice(["US", "GB", "AE", "DE"], n_merchants, p=[.5, .2, .2, .1])
    m_lat, m_lon = rng.uniform(-50, 60, n_merchants), rng.uniform(-120, 120, n_merchants)
    t0 = pd.Timestamp("2025-01-01")
    rows = []

    def add(ts, c, card, dev, m, typ, amt, label):
        rows.append((t0 + pd.to_timedelta(int(ts), unit="s"), c, card, dev, f"10.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(0,255)}",
                     mer[m], m_cat[m], m_cty[m], f"city{m % 40}", m_lat[m], m_lon[m], typ, round(float(amt), 2), "USD", int(label != "none"), label))

    horizon = 90 * 24 * 3600
    ci = rng.integers(0, n_customers, n_tx)
    for k in range(n_tx):
        c = ci[k]
        add(rng.integers(0, horizon), cust[c], f"CARD{c:05d}", f"DEV{c:05d}", rng.integers(0, n_merchants),
            rng.choice(["purchase", "transfer", "withdrawal"], p=[.7, .2, .1]), rng.lognormal(4, 1), "none")
    for c in rng.choice(n_customers, 10, replace=False):                       # card testing: burst of tiny amounts
        start = rng.integers(0, horizon - 7200)
        for _ in range(15):
            add(start + rng.integers(0, 3600), cust[c], f"CARD{c:05d}", f"DEV{c:05d}", rng.integers(0, n_merchants), "purchase", rng.uniform(1, 5), "card_testing")
    for c in rng.choice(n_customers, 20, replace=False):                       # geo anomaly: far-away merchant
        far = int(np.argmax(np.abs(m_lat - m_lat.mean()) + np.abs(m_lon - m_lon.mean())))
        add(rng.integers(0, horizon), cust[c], f"CARD{c:05d}", f"DEV{c:05d}", far, "purchase", rng.lognormal(5, .5), "geo_anomaly")
    for j, c in enumerate(rng.choice(n_customers, 8, replace=False)):         # account takeover: new device, big amounts
        start = rng.integers(0, horizon - 7200)
        for _ in range(3):
            add(start + rng.integers(0, 3600), cust[c], f"CARD{c:05d}", f"DEV_ATO_{j}", rng.integers(0, n_merchants), "transfer", rng.lognormal(4, 1) * 8, "account_takeover")
    for r in range(n_rings):                                                   # laundering rings share a card AND a device
        members = rng.choice(n_customers, ring_size, replace=False)
        hubs = rng.integers(0, n_merchants, 3)
        start = rng.integers(0, horizon - 5 * 86400)
        for rep in range(6):
            for c in members:
                add(start + rep * 3600 * 6 + rng.integers(0, 3600), cust[c], f"RINGCARD_{r}", f"RINGDEV_{r}", rng.choice(hubs), "transfer", rng.lognormal(6, .3), "money_laundering_ring")
    df = pd.DataFrame(rows, columns=["timestamp", "customer_id", "card_id", "device_id", "ip_address", "merchant_id", "merchant_category",
                                     "merchant_country", "merchant_city", "merchant_latitude", "merchant_longitude", "transaction_type",
                                     "amount", "currency", "is_fraud", "fraud_type"])
    df["timestamp"] = df["timestamp"].astype(str)
    df.insert(0, "transaction_id", np.arange(len(df)))
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def make_synthetic_aml(n_tx=60_000, n_accounts=15_000, n_banks=40, per_pattern=14, seed=0):
    """AMLworld-style mock: returns (transactions DataFrame with the raw AMLworld columns, pattern-file text).
    Eight laundering pattern types are injected. Values are invented: code-path tests only."""
    rng = np.random.default_rng(seed)
    acct = [f"8{i:07X}" for i in range(n_accounts)]
    bank = [f"{b:03d}" for b in range(n_banks)]
    a_bank = {a: bank[int(rng.integers(0, n_banks))] for a in acct}
    cur = ["US Dollar", "Euro", "Yuan", "Rupee", "Yen"]
    fmts = ["ACH", "Cheque", "Credit Card", "Wire", "Reinvestment", "Bitcoin", "Cash"]
    t0 = pd.Timestamp("2022-09-01 00:00")
    horizon = 17 * 24 * 60
    rows, text = [], []

    def tx(minute, a, b, amt, label, fmt=None):
        c = cur[int(rng.integers(0, len(cur)))]
        return [(t0 + pd.Timedelta(minutes=int(minute))).strftime("%Y/%m/%d %H:%M"), a_bank[a], a, a_bank[b], b,
                round(float(amt), 2), c, round(float(amt), 2), c, fmt or fmts[int(rng.integers(0, len(fmts)))], label]

    pop = rng.zipf(1.6, n_tx) % n_accounts
    for k in range(n_tx):
        a = acct[int(pop[k])]; b = acct[int(rng.integers(0, n_accounts))]
        if a != b:
            rows.append(tx(rng.integers(0, horizon), a, b, rng.lognormal(7, 1.5), 0))

    def block(name, txs):
        text.append(f"BEGIN LAUNDERING ATTEMPT - {name}")
        for r in txs:
            rows.append(r); text.append(",".join(str(x) for x in r))
        text.append(f"END LAUNDERING ATTEMPT - {name.split(':')[0]}")

    pick = lambda n: [acct[int(i)] for i in rng.choice(n_accounts, n, replace=False)]
    for _ in range(per_pattern):
        amt = float(rng.lognormal(10, .5))
        s, ts = pick(1)[0], pick(int(rng.integers(5, 9)))
        m = int(rng.integers(0, horizon - 600)); block("FAN-OUT", [tx(m + i * 20, s, t, amt / len(ts), 1) for i, t in enumerate(ts)])
        ss, t = pick(int(rng.integers(5, 9))), pick(1)[0]
        m = int(rng.integers(0, horizon - 600)); block("FAN-IN", [tx(m + i * 20, x, t, amt / len(ss), 1) for i, x in enumerate(ss)])
        ring = pick(int(rng.integers(3, 6)))
        m = int(rng.integers(0, horizon - 600)); block("CYCLE: Max 5 hops", [tx(m + i * 60, ring[i], ring[(i + 1) % len(ring)], amt * (0.97 ** i), 1) for i in range(len(ring))])
        src, hub, dst = pick(4), pick(1)[0], pick(4)
        m = int(rng.integers(0, horizon - 600)); block("GATHER-SCATTER", [tx(m + i * 15, x, hub, amt / 4, 1) for i, x in enumerate(src)] +
              [tx(m + 200 + i * 15, hub, y, amt / 4 * 0.98, 1) for i, y in enumerate(dst)])
        o, mid, sink = pick(1)[0], pick(4), pick(1)[0]
        m = int(rng.integers(0, horizon - 600)); block("SCATTER-GATHER", [tx(m + i * 15, o, x, amt / 4, 1) for i, x in enumerate(mid)] +
              [tx(m + 200 + i * 15, x, sink, amt / 4 * 0.98, 1) for i, x in enumerate(mid)])
        l, r = pick(3), pick(3)
        m = int(rng.integers(0, horizon - 600)); block("BIPARTITE", [tx(m + (i * 3 + j) * 10, x, y, amt / 9, 1) for i, x in enumerate(l) for j, y in enumerate(r)])
        l1, l2, l3 = pick(3), pick(3), pick(3)
        m = int(rng.integers(0, horizon - 600)); block("STACK", [tx(m + (i * 3 + j) * 10, x, y, amt / 9, 1) for i, x in enumerate(l1) for j, y in enumerate(l2)] +
              [tx(m + 400 + (i * 3 + j) * 10, x, y, amt / 10, 1) for i, x in enumerate(l2) for j, y in enumerate(l3)])
        walk = pick(6)
        m = int(rng.integers(0, horizon - 600)); block("RANDOM", [tx(m + i * 45, walk[i], walk[i + 1], amt * (0.95 ** i), 1) for i in range(5)])
    for _ in range(per_pattern):                               # laundering rows outside any listed pattern
        a, b = pick(2); rows.append(tx(rng.integers(0, horizon), a, b, rng.lognormal(9, .5), 1))
    df = pd.DataFrame(rows, columns=["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Received",
                                     "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"])
    df = df.assign(_t=pd.to_datetime(df.Timestamp)).sort_values("_t", kind="stable").drop(columns="_t").reset_index(drop=True)
    return df, "\n".join(text)
