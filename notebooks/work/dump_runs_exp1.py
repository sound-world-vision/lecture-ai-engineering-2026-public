"""Dump all runs of MLflow experiment id 1 as JSON (requested by peer session)."""
import json

import mlflow

client = mlflow.MlflowClient()
runs = client.search_runs(["1"])
rows = [
    {
        "info": {
            "experiment_id": r.info.experiment_id,
            "run_id": r.info.run_id,
            "artifact_uri": r.info.artifact_uri,
            "status": r.info.status,
            "start_time": r.info.start_time,
            "end_time": r.info.end_time,
            "lifecycle_stage": r.info.lifecycle_stage,
            "run_name": r.info.run_name,
            "user_id": r.info.user_id,
        },
        "metrics": r.data.metrics,
        "params": r.data.params,
        "tags": r.data.tags,
    }
    for r in runs
]
with open("notebooks/work/runs_exp1_cli.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(f"wrote {len(rows)} runs to notebooks/work/runs_exp1_cli.json")
