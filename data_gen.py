"""Synthetic digital-payment transaction generator (Transaction Data stage)."""
import numpy as np
import pandas as pd

MERCHANTS = {  # merchant category -> inherent risk (0..1)
    "grocery": 0.10, "fuel": 0.15, "restaurant": 0.15, "utilities": 0.10,
    "pharmacy": 0.10, "travel": 0.40, "electronics": 0.60,
    "jewelry": 0.75, "gift_cards": 0.85, "crypto": 0.90,
}
SAFE = [m for m, r in MERCHANTS.items() if r <= 0.15]
RISKY = [m for m, r in MERCHANTS.items() if r >= 0.6]


def _offset(rng, lat, lon, dist_km):
    th = rng.uniform(0, 2 * np.pi)
    return (lat + dist_km * np.cos(th) / 111.0,
            lon + dist_km * np.sin(th) / (111.0 * np.cos(np.radians(lat))))


def generate(n_users=300, n_txn=18000, days=60, warmup_days=12, seed=7):
    rng = np.random.default_rng(seed)
    users = [dict(id=f"U{u:04d}", lat=rng.uniform(8, 28), lon=rng.uniform(72, 88),
                  mu=rng.normal(6.4, 0.5), sd=rng.uniform(0.35, 0.6),
                  hc=rng.uniform(10, 20), hsd=rng.uniform(2, 3.5),
                  devices=[f"D{u:04d}-{k}" for k in range(rng.integers(1, 3))])
             for u in range(n_users)]
    rows = []

    def add(u, ts, amount, lat, lon, dev, merch, fraud):
        rows.append(dict(user_id=u["id"], ts=ts, hour=ts % 24, amount=round(float(amount), 2),
                         lat=lat, lon=lon, device_id=dev, merchant=merch, is_fraud=int(fraud)))

    # ---- legitimate behaviour
    for _ in range(n_txn):
        u = users[rng.integers(n_users)]
        hour = rng.normal(u["hc"], u["hsd"]) % 24
        ts = rng.integers(0, days) * 24 + hour
        d = rng.uniform(50, 300) if rng.random() < 0.05 else rng.exponential(6)
        lat, lon = _offset(rng, u["lat"], u["lon"], d)
        dev = rng.choice(u["devices"]) if rng.random() > 0.035 else f"NEW-{rng.integers(1e9)}"
        mu = u["mu"] + (1.5 if rng.random() < 0.02 else 0)
        merch = rng.choice(SAFE) if rng.random() < 0.88 else rng.choice(list(MERCHANTS))
        amt = rng.lognormal(mu, u["sd"])
        add(u, ts, amt, lat, lon, dev, merch, False)
        if rng.random() < 0.04:  # harmless legit burst
            add(u, ts + rng.uniform(0.08, 0.3), rng.lognormal(u["mu"], u["sd"]), lat, lon, dev, merch, False)

    # ---- fraud events (account takeover / card-not-present patterns)
    for _ in range(int(n_txn * 0.04)):
        u = users[rng.integers(n_users)]
        k = rng.choice([1, 2, 3, 4, 5], p=[.12, .28, .30, .20, .10])
        traits = set(rng.choice(["amount", "hour", "loc", "device", "merchant"], size=k, replace=False))
        hour = rng.uniform(0, 5) if "hour" in traits else rng.normal(u["hc"], u["hsd"]) % 24
        t0 = rng.integers(warmup_days, days) * 24 + hour
        lat, lon = _offset(rng, u["lat"], u["lon"], rng.uniform(200, 2000) if "loc" in traits else rng.exponential(8))
        dev = f"NEW-{rng.integers(1e9)}" if ("device" in traits and rng.random() < .9) else rng.choice(u["devices"])
        merch = rng.choice(RISKY) if "merchant" in traits else rng.choice(SAFE)
        burst = 1 if rng.random() < 0.5 else rng.integers(2, 5)
        t = t0
        for _b in range(burst):
            mu = u["mu"] + (rng.uniform(1.2, 2.6) if "amount" in traits else 0)
            add(u, t, rng.lognormal(mu, u["sd"]), lat, lon, dev, merch, True)
            t += rng.uniform(0.05, 0.25)

    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return df


if __name__ == "__main__":
    d = generate()
    print(d.shape, "fraud rate:", round(d.is_fraud.mean(), 4))
