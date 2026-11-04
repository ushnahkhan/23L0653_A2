import os
import time

import mlflow
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

MODEL_NAME = "23L0653_A2_Ops_VisionGuard"
EXPERIMENT = "23L0653_A2_Ops_visionguard_comparison"
VERSIONS = [("baseline", "challenger", "MobileNetV2 224 px - tuned baseline"),
            ("img_160", "champion", "MobileNetV2 160 px - best val AUC in Task 3.5")]

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = MlflowClient()


def latest_run(run_name):
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs([exp.experiment_id],
                                filter_string=f"tags.mlflow.runName = '{run_name}'",
                                order_by=["start_time DESC"], max_results=1)
    if not runs:
        raise SystemExit(f"Run '{run_name}' not found in {EXPERIMENT}")
    return runs[0]


# 1. registered model (container for all versions)
try:
    client.create_registered_model(
        MODEL_NAME, description="VisionGuard: AI-generated vs real image classifier (sigmoid = P(AI))")
    print(f"Created registered model '{MODEL_NAME}'")
except MlflowException:
    print(f"Registered model '{MODEL_NAME}' already exists - adding new versions")

# 2. one version per run (source = the run's logged model folder)
for run_name, alias, note in VERSIONS:
    run = latest_run(run_name)
    p, m, t = run.data.params, run.data.metrics, run.data.tags
    mv = client.create_model_version(
        name=MODEL_NAME, source=f"{run.info.artifact_uri}/model", run_id=run.info.run_id,
        description=note,
        tags={"img_size": p["img_size"], "backbone": p["backbone"],
                "threshold": f"{m['sel_threshold']:.2f}", "val_auc": f"{m['sel_val_auc']:.4f}",
                "git_commit": t.get("git_commit", ""), "dvc_data_md5": t.get("dvc_data_md5", ""),
                "source_run": run_name})
    for _ in range(30):                                       # wait until READY
        if client.get_model_version(MODEL_NAME, mv.version).status == "READY":
            break
        time.sleep(1)

    # 3. alias (fallback to a version tag if the server has no alias support)
    try:
        client.set_registered_model_alias(MODEL_NAME, alias, mv.version)
        how = "alias"
    except MlflowException as e:
        client.set_model_version_tag(MODEL_NAME, mv.version, "alias", alias)
        how = f"tag (alias API unsupported: {type(e).__name__})"
    print(f"Version {mv.version}: run '{run_name}' | img_size {p['img_size']} | "
            f"threshold {m['sel_threshold']:.2f} | val_auc {m['sel_val_auc']:.4f} -> '{alias}' ({how})")

print(f"\nOpen DagsHub -> Models -> {MODEL_NAME}")