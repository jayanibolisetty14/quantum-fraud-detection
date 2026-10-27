"""Behavioural score, transaction risk, profile risk, risk fusion and explanations."""
import numpy as np

BEHAV_W = dict(amount=.25, time=.15, location=.25, device=.25, frequency=.10)
FUSION_W = dict(quantum=.40, behavioral=.30, transaction=.15, profile=.15)


def behavioral_score(c):
    weighted = sum(BEHAV_W[k] * c[k] for k in BEHAV_W)
    return float(np.clip(0.6 * weighted + 0.4 * max(c.values()) * 0.8, 0, 1))


def transaction_risk(t, a):
    amt = t["amount"]
    band = 1.0 if amt > 25000 else 0.6 if amt > 10000 else 0.3 if amt > 5000 else 0.0
    night = 1.0 if (a["raw"]["hour"] < 5 or a["raw"]["hour"] >= 23.5) else 0.0
    return float(0.5 * a["raw"]["merchant_risk"] + 0.3 * band + 0.2 * night)


def profile_risk(prof):
    maturity = min(1.0, prof.n / 30)
    return float(0.6 * (1 - maturity) * 0.5 + 0.4 * min(1.0, prof.flag_ema * 3))


def fuse(q, b, t, p, w=FUSION_W):
    return float(np.clip(w["quantum"] * q + w["behavioral"] * b + w["transaction"] * t + w["profile"] * p, 0, 1))


def decide(score, t_med, t_high):
    return "ALLOW" if score < t_med else "VERIFY" if score < t_high else "HOLD"


def explain(t, a, q, thr=0.4):
    c, r = a["comps"], a["raw"]
    out = []
    if c["device"] >= thr:
        out.append(("New device", c["device"], "Device not seen in this user's trusted history"))
    if c["amount"] >= thr:
        out.append(("Unusual amount", c["amount"],
                    f"Rs {t['amount']:,.0f} is {t['amount'] / r['typical_amount']:.1f}x the user's typical Rs {r['typical_amount']:,.0f}"))
    if c["time"] >= thr:
        out.append(("Unusual time", c["time"], f"{int(r['hour']):02d}:00 is an hour this user rarely transacts"))
    if c["location"] >= thr:
        out.append(("Location deviation", c["location"], f"{r['dist_km']:,.0f} km from the user's usual location"))
    if c["frequency"] >= thr:
        out.append(("High frequency", c["frequency"], f"{r['n1h']} other attempts in the last hour"))
    if r["merchant_risk"] >= 0.6:
        out.append(("Risky merchant", r["merchant_risk"], f"'{t['merchant']}' is a high-risk category"))
    if q >= 0.5:
        out.append(("Fraud similarity", q, f"Quantum kernel similarity to known fraud patterns: {q:.0%}"))
    out.sort(key=lambda x: -x[1])
    return [dict(factor=f, severity=round(float(s), 3), detail=d) for f, s, d in out]
