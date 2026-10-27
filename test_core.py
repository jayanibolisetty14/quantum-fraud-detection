import numpy as np
from backend.quantum import zz_states, kernel_matrix
from backend.detector import FraudDetector


def test_kernel_properties():
    X = np.random.default_rng(0).uniform(0, np.pi, (20, 4))
    S = zz_states(X); K = kernel_matrix(S, S)
    assert np.allclose(np.linalg.norm(S, axis=1), 1)          # normalised states
    assert np.allclose(np.diag(K), 1) and np.allclose(K, K.T)  # k(x,x)=1, symmetric
    assert np.linalg.eigvalsh(K).min() > -1e-9                 # positive semi-definite
    assert K.min() >= 0 and K.max() <= 1 + 1e-9


def test_takeover_scores_higher_than_normal():
    import math
    det = FraudDetector.load("models/detector.pkl")
    uid = next(u for u, p in det.store.items() if p.n >= 20); p = det.store[uid]
    base = dict(user_id=uid, lat=p.lat, lon=p.lon, device_id=sorted(p.devices)[0], amount=math.exp(p.mean),
                hour=float(p.hist.argmax()), merchant="grocery", ts=1e6)
    bad = {**base, "device_id": "NEW-X", "lat": p.lat + 9, "lon": p.lon + 6, "amount": math.exp(p.mean + 2.2),
           "hour": 3.0, "merchant": "crypto", "ts": 1e6 + 5}
    a = det.score(base, learn=False); b = det.score(bad, learn=False)
    assert b["risk_score"] > a["risk_score"] and b["decision"] == "HOLD" and a["decision"] == "ALLOW"
    assert any(r["factor"] == "New device" for r in b["reasons"])
