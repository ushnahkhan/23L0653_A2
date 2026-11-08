"""Download the champion model from the MLflow registry into model_store/ (used by CI before docker build).
Needs MLFLOW_TRACKING_URI / _USERNAME / _PASSWORD as environment variables (GitHub secrets in CI)."""
import json
import os
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL_NAME, ALIAS = "23L0653_A2_Ops_VisionGuard", "champion"
STORE = Path(os.getenv("VISIONGUARD_MODEL_DIR", "model_store"))

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = MlflowClient()
try:
    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    uri = f"models:/{MODEL_NAME}@{ALIAS}"
except MlflowException:
    mv = client.search_model_versions(f"name = '{MODEL_NAME}' and tags.alias = '{ALIAS}'")[0]
    uri = f"models:/{MODEL_NAME}/{mv.version}"

STORE.mkdir(parents=True, exist_ok=True)
mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(STORE))
meta = {"model_name": MODEL_NAME, "alias": ALIAS, "version": mv.version, "run_id": mv.run_id,
        "img_size": int(mv.tags["img_size"]), "threshold": float(mv.tags["threshold"]),
        "source_run": mv.tags.get("source_run", "")}
(STORE / "model_meta.json").write_text(json.dumps(meta, indent=2))
print(f"Fetched {uri} (version {mv.version}) -> {STORE}")