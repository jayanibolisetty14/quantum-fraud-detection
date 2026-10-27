"""End-to-end hybrid quantum-classical fraud detector following the architecture diagram."""
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from .features import analyze, FEATURE_NAMES
from .profile import ProfileStore
from .quantum import QuantumKernelClassifier
from .scoring import (behavioral_score, transaction_risk, profile_risk, fuse,
                      decide, explain, FUSION_W)


def best_f1_threshold(s, y):
    grid = np.unique(np.round(np.quantile(s, np.linspace(0.5, 0.995, 120)), 4))
    f = [precision_recall_fscore_support(y, s >= g, average="binary", zero_division=0)[2] for g in grid]
    return float(grid[int(np.argmax(f))])


class FraudDetector:
    def __init__(self, n_train_per_class=150, backend="numpy", seed=7):
        self.n_pc, self.backend, self.seed = n_train_per_class, backend, seed
        self.store = ProfileStore()
        self.t_med, self.t_high = 0.35, 0.60
        self.metrics = {}

    # ---------------- training ----------------
    def fit(self, df):
        df = df.sort_values("ts").reset_index(drop=True)
        t_tr, t_va = df.ts.quantile(.60), df.ts.quantile(.75)
        X, y = [], []
        for t in df[df.ts <= t_tr].to_dict("records"):          # label-aware profile building
            prof = self.store.get_profile(t["user_id"])
            a = analyze(t, prof)
            if prof.n >= 5:
                X.append(a["x"]); y.append(t["is_fraud"])
            prof.record_attempt(t["ts"])
            if not t["is_fraud"]:
                prof.update(t)
        X, y = np.array(X), np.array(y)

        # Preprocessing + Feature Reduction (scale -> select -> PCA(4 qubits) -> [0, pi])
        self.prep = Pipeline([("scale", StandardScaler()),
                              ("select", SelectKBest(f_classif, k=6)),
                              ("pca", PCA(4, random_state=self.seed)),
                              ("angle", MinMaxScaler((0, np.pi), clip=True))]).fit(X, y)
        Z = self.prep.transform(X)

        rng = np.random.default_rng(self.seed)
        fr, lg = np.where(y == 1)[0], np.where(y == 0)[0]
        idx = np.r_[rng.choice(fr, min(self.n_pc, len(fr)), replace=False),
                    rng.choice(lg, self.n_pc, replace=False)]
        self.qclf = QuantumKernelClassifier(backend=self.backend, seed=self.seed).tune(Z[idx], y[idx]).fit(Z[idx], y[idx])
        self.csvm = SVC(kernel="rbf", C=4.0, gamma="scale", probability=True,
                        class_weight="balanced", random_state=self.seed).fit(Z[idx], y[idx])
        self.gbm = GradientBoostingClassifier(random_state=self.seed).fit(X, y)
        self.n_train, self.n_train_fraud = len(idx), int(y[idx].sum())

        # Validation stream -> tune thresholds ; test stream -> report
        val = self._stream(df[(df.ts > t_tr) & (df.ts <= t_va)])
        self.thr = {k: best_f1_threshold(val[k], val["y"]) for k in
                    ["quantum", "classical_svm", "classical_gbm", "behavioral", "final", "final_classical"]}
        f, yv = val["final"], val["y"]
        t_f1 = self.thr["final"]
        hi = [g for g in np.linspace(t_f1, 0.95, 80) if (f >= g).sum() >= 10 and yv[f >= g].mean() >= 0.9]
        self.t_high = float(hi[0]) if hi else t_f1 + 0.1                      # HOLD: high precision
        lo = [g for g in np.linspace(0.1, self.t_high - 0.05, 80) if np.mean(f[yv == 0] >= g) <= 0.03]
        self.t_med = float(lo[0]) if lo else 0.75 * t_f1                      # VERIFY: <=3% legit friction
        test = self._stream(df[df.ts > t_va])
        self.metrics = self._report(test)
        return self

    def _stream(self, d):
        keys = ["quantum", "classical_svm", "classical_gbm", "behavioral", "transaction",
                "profile", "final", "final_classical", "y", "decision"]
        log = {k: [] for k in keys}
        for t in d.to_dict("records"):
            r = self.score(t, learn=True, verified=not t["is_fraud"])
            s = r["scores"]
            log["quantum"].append(s["quantum"]); log["classical_svm"].append(s["classical_svm"])
            log["classical_gbm"].append(s["classical_gbm"]); log["behavioral"].append(s["behavioral"])
            log["transaction"].append(s["transaction"]); log["profile"].append(s["profile"])
            log["final"].append(r["risk_score"])
            log["final_classical"].append(fuse(s["classical_svm"], s["behavioral"], s["transaction"], s["profile"]))
            log["y"].append(t["is_fraud"]); log["decision"].append(r["decision"])
        return {k: np.array(v) for k, v in log.items()}

    def _report(self, L):
        y = L["y"]
        names = {"quantum": "Quantum kernel SVM (4 qubits)", "classical_svm": "Classical RBF SVM (same 4 features)",
                 "classical_gbm": "Classical gradient boosting (8 raw features)",
                 "behavioral": "Behavioural score only",
                 "final_classical": "Fusion with classical SVM instead of quantum",
                 "final": "Hybrid fusion (quantum + behavioural + risk)"}
        rows = []
        for k, label in names.items():
            p, r, f, _ = precision_recall_fscore_support(y, L[k] >= self.thr[k], average="binary", zero_division=0)
            rows.append(dict(model=label, roc_auc=round(roc_auc_score(y, L[k]), 4), precision=round(p, 4),
                             recall=round(r, 4), f1=round(f, 4)))
        dec = L["decision"]; legit = y == 0
        return dict(
            test_transactions=int(len(y)), test_fraud=int(y.sum()),
            quantum_train_samples=self.n_train, quantum_kernel_scale=self.qclf.scale, quantum_C=self.qclf.C, quantum_cv_auc=round(float(self.qclf.cv_auc), 4), quantum_train_fraud=self.n_train_fraud,
            models=rows,
            thresholds=dict(verify=round(self.t_med, 3), hold=round(self.t_high, 3)),
            decisions=dict(
                fraud_caught_pct=round(float(np.mean(dec[y == 1] != "ALLOW")) * 100, 1),
                fraud_held_pct=round(float(np.mean(dec[y == 1] == "HOLD")) * 100, 1),
                legit_friction_pct=round(float(np.mean(dec[legit] != "ALLOW")) * 100, 2)))

    # ---------------- inference ----------------
    def score(self, t, learn=True, verified=None):
        prof = self.store.get_profile(t["user_id"])
        a = analyze(t, prof)
        Z = self.prep.transform(a["x"][None])
        q = float(self.qclf.predict_proba(Z)[0])
        c = float(self.csvm.predict_proba(Z)[0, 1])
        g = float(self.gbm.predict_proba(a["x"][None])[0, 1])
        b, tr, pr = behavioral_score(a["comps"]), transaction_risk(t, a), profile_risk(prof)
        final = fuse(q, b, tr, pr)
        decision = decide(final, self.t_med, self.t_high)
        reasons = explain(t, a, q) if decision != "ALLOW" else []
        prof.record_attempt(t["ts"])
        if learn:
            prof.note_outcome(decision != "ALLOW")
            if decision == "ALLOW" or verified:
                prof.update(t)
        return dict(risk_score=round(final, 4), decision=decision,
                    risk_level={"ALLOW": "LOW", "VERIFY": "MEDIUM", "HOLD": "HIGH"}[decision],
                    scores=dict(quantum=q, classical_svm=c, classical_gbm=g, behavioral=b,
                                transaction=tr, profile=pr),
                    components=a["comps"], reasons=reasons,
                    fusion_weights=FUSION_W, quantum_features=[round(float(v), 4) for v in Z[0]])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
