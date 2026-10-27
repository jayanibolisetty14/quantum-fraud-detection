"""Quantum Fraud Similarity Engine: ZZFeatureMap -> fidelity quantum kernel -> kernel SVM.

States are simulated exactly with numpy (no Qiskit needed). The circuit is the same
definition as qiskit.circuit.library.ZZFeatureMap (reps=2, linear entanglement);
`verify_against_qiskit` checks that when Qiskit is installed, and backend="qiskit"
uses Qiskit's Statevector instead.
"""
import numpy as np
from sklearn.svm import SVC


def _bits(n):
    return ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)  # qubit 0 = LSB


def zz_states(X, reps=2):
    X = np.atleast_2d(np.asarray(X, float))
    n = X.shape[1]
    bits = _bits(n)
    H = (-1.0) ** (bits @ bits.T) / np.sqrt(2 ** n)  # H on every qubit
    pairs = [(i, i + 1) for i in range(n - 1)]       # linear entanglement
    psi = np.full((len(X), 2 ** n), 1 / np.sqrt(2 ** n), dtype=complex)  # H|0..0>
    for r in range(reps):
        if r:
            psi = psi @ H.T
        phase = 2 * X @ bits.T                               # P(2 x_i) on each qubit
        for i, j in pairs:                                   # CX - P(2(pi-xi)(pi-xj)) - CX
            theta = 2 * (np.pi - X[:, i]) * (np.pi - X[:, j])
            phase = phase + theta[:, None] * np.logical_xor(bits[:, i], bits[:, j])[None, :]
        psi = psi * np.exp(1j * phase)
    return psi


def qiskit_states(X, reps=2):
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.quantum_info import Statevector
    X = np.atleast_2d(X)
    fm = ZZFeatureMap(X.shape[1], reps=reps, entanglement="linear")
    return np.array([Statevector(fm.assign_parameters(x)).data for x in X])


def kernel_matrix(A, B):
    """Fidelity kernel k(x,y) = |<psi(x)|psi(y)>|^2 from state matrices."""
    return np.abs(A @ B.conj().T) ** 2


def verify_against_qiskit(n=6, seed=0):
    X = np.random.default_rng(seed).uniform(0, np.pi, (n, 4))
    a, b = zz_states(X), qiskit_states(X)
    return float(np.max(np.abs(kernel_matrix(a, a) - kernel_matrix(b, b))))


def circuit_text(n_qubits=4, reps=2):
    """ASCII drawing of the feature map (needs Qiskit; used by the Quantum Lab view)."""
    try:
        from qiskit.circuit.library import ZZFeatureMap
        return str(ZZFeatureMap(n_qubits, reps=reps, entanglement="linear").decompose().draw("text"))
    except Exception:
        return "Install qiskit to render the circuit: pip install qiskit"


class QuantumKernelClassifier:
    def __init__(self, reps=2, C=4.0, scale=1.0, backend="numpy", seed=7):
        self.reps, self.C, self.scale, self.backend, self.seed = reps, C, scale, backend, seed

    def _states(self, X):
        X = np.asarray(X) * self.scale   # scale acts as the kernel bandwidth
        return (qiskit_states if self.backend == "qiskit" else zz_states)(X, self.reps)

    def tune(self, X, y, scales=(0.15, 0.25, 0.4, 0.6, 0.8, 1.0), Cs=(1.0, 4.0, 16.0), folds=5):
        """Pick (scale, C) by stratified CV ROC-AUC on the training set only."""
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        y = np.asarray(y); best = (-1, self.scale, self.C)
        for sc in scales:
            self.scale = sc
            S = self._states(X); K = kernel_matrix(S, S)
            for C in Cs:
                oof = np.zeros(len(y))
                for tr, te in StratifiedKFold(folds, shuffle=True, random_state=self.seed).split(X, y):
                    m = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(K[np.ix_(tr, tr)], y[tr])
                    oof[te] = m.decision_function(K[np.ix_(te, tr)])
                auc = roc_auc_score(y, oof)
                if auc > best[0]:
                    best = (auc, sc, C)
        self.cv_auc, self.scale, self.C = best
        return self

    def fit(self, X, y):
        self.S = self._states(X)
        self.clf = SVC(kernel="precomputed", C=self.C, probability=True,
                       class_weight="balanced", random_state=self.seed)
        self.clf.fit(kernel_matrix(self.S, self.S), y)
        return self

    def predict_proba(self, X):
        K = kernel_matrix(self._states(X), self.S)
        return self.clf.predict_proba(K)[:, 1]
