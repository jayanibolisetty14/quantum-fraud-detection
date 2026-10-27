"""python train.py  -> generates data, trains the pipeline, evaluates, saves models/."""
import json, time, os
from backend.data_gen import generate
from backend.detector import FraudDetector

if __name__ == "__main__":
    t0 = time.time()
    df = generate()
    print(f"data: {len(df)} txns, fraud rate {df.is_fraud.mean():.3%}")
    det = FraudDetector().fit(df)
    os.makedirs("models", exist_ok=True)
    det.save("models/detector.pkl")
    json.dump(det.metrics, open("models/metrics.json", "w"), indent=2)
    print(json.dumps(det.metrics, indent=2))
    print(f"done in {time.time() - t0:.0f}s")
