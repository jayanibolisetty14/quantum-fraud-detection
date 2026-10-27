"""FastAPI service.  Run:  uvicorn backend.api:app --reload --port 8000"""
import json, math, os, time, uuid
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .detector import FraudDetector
from .data_gen import MERCHANTS
from .quantum import circuit_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = FastAPI(title="Quantum-Enhanced Digital Payment Fraud Detection")
det: FraudDetector = FraudDetector.load(os.path.join(ROOT, "models", "detector.pkl"))
pending = {}


class Txn(BaseModel):
    user_id: str
    amount: float
    lat: float
    lon: float
    device_id: str
    merchant: str = "grocery"
    hour: Optional[float] = None          # local hour 0-24 (default: derived from timestamp)
    timestamp: Optional[str] = None       # ISO-8601, default now


@app.get("/api/health")
def health():
    return {"status": "ok", "users": len(det.store), "quantum_backend": det.qclf.backend}


@app.get("/api/metrics")
def metrics():
    return det.metrics


@app.get("/api/merchants")
def merchants():
    return MERCHANTS


@app.get("/api/quantum/circuit")
def circuit():
    return {"qubits": 4, "feature_map": "ZZFeatureMap(reps=2, entanglement=linear)",
            "kernel": "fidelity |<psi(x)|psi(y)>|^2", "diagram": circuit_text()}


@app.get("/api/users")
def users(limit: int = 30):
    return [u for u, p in det.store.items() if p.n >= 20][:limit]


@app.get("/api/demo/{user_id}")
def demo(user_id: str):
    if user_id not in det.store:
        raise HTTPException(404, "unknown user")
    p = det.store[user_id]
    base = dict(user_id=user_id, lat=p.lat, lon=p.lon, device_id=sorted(p.devices)[0],
                amount=round(math.exp(p.mean)), hour=float(p.hist.argmax()), merchant="grocery")
    return {
        "normal purchase": base,
        "new device only": {**base, "device_id": "NEW-DEMO-1"},
        "account takeover": {**base, "device_id": "NEW-DEMO-2", "lat": p.lat + 9, "lon": p.lon + 6,
                             "amount": round(math.exp(p.mean + 2.2)), "hour": 3.0, "merchant": "crypto"},
    }


@app.post("/api/score")
def score(t: Txn):
    if t.merchant not in MERCHANTS:
        raise HTTPException(422, f"merchant must be one of {list(MERCHANTS)}")
    dt = datetime.fromisoformat(t.timestamp) if t.timestamp else datetime.now()
    txn = dict(user_id=t.user_id, amount=max(t.amount, 1.0), lat=t.lat, lon=t.lon,
               device_id=t.device_id, merchant=t.merchant, ts=dt.timestamp() / 3600,
               hour=t.hour if t.hour is not None else dt.hour + dt.minute / 60)
    r = det.score(txn, learn=True)
    r["txn_id"] = uuid.uuid4().hex[:10]
    if r["decision"] != "ALLOW":
        pending[r["txn_id"]] = txn
    return r


@app.post("/api/confirm/{txn_id}")
def confirm(txn_id: str):
    """User verified a VERIFY-ed transaction -> the adaptive profile learns from it."""
    txn = pending.pop(txn_id, None)
    if txn is None:
        raise HTTPException(404, "no pending transaction")
    det.store.get_profile(txn["user_id"]).update(txn)
    return {"status": "profile updated"}


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "static")), name="static")
