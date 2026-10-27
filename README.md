# Quantum-Enhanced Digital Payment Fraud Detection

Hybrid quantum-classical fraud detector that implements the project architecture one-to-one.

| Architecture block | Code |
|---|---|
| Transaction data | `backend/data_gen.py` (synthetic users, devices, locations, merchants, fraud bursts) |
| Data preprocessing / Feature reduction | `backend/detector.py` -> scale, SelectKBest(6), PCA(4 qubits), angle map to [0, pi] |
| Quantum fraud similarity engine | `backend/quantum.py` -> ZZFeatureMap (reps=2, linear) -> fidelity kernel -> kernel SVM |
| Behavioural analysis | `backend/features.py` + `backend/scoring.py` (amount, time, location, device, frequency) |
| Adaptive user risk profile | `backend/profile.py` (online mean/variance, hour histogram, known devices, moving home, flag rate) |
| Risk fusion engine | `backend/scoring.py::fuse` = 0.40 quantum + 0.30 behavioural + 0.15 transaction + 0.15 profile |
| Low / Medium / High -> Allow / Verify / Hold | thresholds tuned on a validation stream (`detector.py`) |
| Explainable fraud alert | `backend/scoring.py::explain` |

## Run
```bash
pip install -r requirements.txt
python train.py                      # ~25 s: data -> train -> evaluate -> models/
uvicorn backend.api:app --port 8000  # dashboard at http://localhost:8000
python -c "import tests.test_core as t; t.test_kernel_properties(); t.test_takeover_scores_higher_than_normal()"
```
The quantum states are simulated exactly with numpy so everything runs without Qiskit. With Qiskit installed:
`from backend.quantum import verify_against_qiskit; verify_against_qiskit()` (should be ~1e-15) and construct
`FraudDetector(backend="qiskit")` to run the kernel through Qiskit's `Statevector`.
Swap in IBM hardware/Aer by replacing `qiskit_states` with a `FidelityQuantumKernel` + sampler.

## Method notes
* Strict time split: 60% train (profiles built, models fit) / 15% validation (thresholds tuned) / 25% test (streamed, profiles adapt online).
* Quantum SVM trains on 300 balanced samples (4 qubits); kernel bandwidth and C picked by 5-fold CV on the training set only.
* The profile only learns from allowed or user-verified transactions, so fraud cannot poison "normal".
* Data is synthetic: results demonstrate the pipeline, not real-world performance. Replace `generate()` with a real dataset
  (columns: user_id, ts, hour, amount, lat, lon, device_id, merchant, is_fraud) to evaluate on real data.
