import os
import subprocess

from dotenv import load_dotenv

from tag_analyzer import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
EXPERIMENT = "tag-analyzer"
MISTAKES_FILE = "mistakes.txt"
UNKNOWN_GIT_COMMIT = "unknown"


def tracking_enabled() -> bool:
    return bool(MLFLOW_TRACKING_URI)


def git_commit() -> str:
    try:
        finished = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
        )
    except OSError:
        return UNKNOWN_GIT_COMMIT
    return finished.stdout.strip() or UNKNOWN_GIT_COMMIT


def log_eval_run(settings: dict, metrics: dict, mistakes: list[str]) -> str | None:
    if not tracking_enabled():
        return None
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.log_params({**settings, "git_commit": git_commit()})
        mlflow.log_metrics(metrics)
        mlflow.log_text("\n".join(mistakes), MISTAKES_FILE)
    return run.info.run_id
