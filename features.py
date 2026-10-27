"""Feature extraction (quantum feature vector + behavioural component scores)."""
import math
import numpy as np
from .data_gen import MERCHANTS
from .profile import haversine_km

FEATURE_NAMES = ["log_amount", "amount_zscore", "hour_sin", "hour_cos",
                 "log_distance_km", "new_device", "txn_last_hour", "merchant_risk"]


def analyze(t, prof):
    hour = float(t["hour"]) % 24
    la = math.log(max(t["amount"], 1.0))
    if prof.n == 0:  # cold start: nothing known yet
        z, dist, new_dev = 0.0, 0.0, 1.0
    else:
        z = (la - prof.mean) / prof.std
        dist = haversine_km(prof.lat, prof.lon, t["lat"], t["lon"])
        new_dev = float(t["device_id"] not in prof.devices)
    n1h = sum(1 for s in prof.recent if 0 < t["ts"] - s <= 1.0)
    merch = MERCHANTS.get(t["merchant"], 0.5)

    h = int(hour)
    sm = 0.25 * np.roll(prof.hist, 1) + 0.5 * prof.hist + 0.25 * np.roll(prof.hist, -1)
    time_s = float(np.clip(1 - sm[h] / (0.3 * sm.max()), 0, 1))

    ang = 2 * math.pi * hour / 24
    x = np.array([la, np.clip(z, -6, 6), math.sin(ang), math.cos(ang),
                  math.log1p(dist), new_dev, n1h, merch])
    comps = dict(
        amount=float(np.clip((z - 1.5) / 3, 0, 1)),
        time=time_s,
        location=float(np.clip(math.log1p(dist / 50) / math.log1p(20), 0, 1)),
        device=new_dev,
        frequency=float(np.clip(n1h / 3, 0, 1)),
    )
    raw = dict(z=z, dist_km=dist, new_device=bool(new_dev), n1h=n1h, merchant_risk=merch,
               hour=hour, typical_amount=math.exp(prof.mean))
    return dict(x=x, comps=comps, raw=raw)
