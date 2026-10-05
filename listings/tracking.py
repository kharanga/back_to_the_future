import hashlib
import subprocess
from pathlib import Path

from listings import ADAPTER_DIR, EVAL_FILE, JUDGE_FILE, PROJECT_ROOT, TRAIN_FILE, VAL_FILE, config
from listings.invariants import INVARIANT_NAMES, invariant_pass_rates

EXPERIMENT = "listings-first-line"
RUN_ID_FILE_NAME = "mlflow_run_id.txt"
ADAPTER_ARTIFACT_PATH = "adapter"
EVAL_TABLE_FILE = "eval_rows.json"
EVAL_TABLE_COLUMNS = ["listing_id", "target", "generated", *INVARIANT_NAMES]
DATASET_HASH_LENGTH = 12
UNKNOWN_GIT_COMMIT = "unknown"
REPORT_TO_MLFLOW = "mlflow"
REPORT_TO_NOTHING = "none"
UNKNOWN_RUN_ID_PARAM = "adapter_run_id_not_found"
BASELINE_RUN_NAME = "shop-baseline"
JUDGE_TABLE_FILE = "judge_rows.json"
LIST_CELL_SEPARATOR = "; "
JUDGE_METRIC_PREFIX = "judge_"
SIMILARITY_METRIC_PREFIX = "similarity_"
RUBRIC_HASH_LENGTH = 12
RUN_NOT_FOUND_ERROR_CODES = {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}


def tracking_enabled() -> bool:
    return bool(config.MLFLOW_TRACKING_URI)


def report_to() -> str:
    return REPORT_TO_MLFLOW if tracking_enabled() else REPORT_TO_NOTHING


def git_commit() -> str:
    try:
        finished = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
        )
    except OSError:
        return UNKNOWN_GIT_COMMIT
    return finished.stdout.strip() or UNKNOWN_GIT_COMMIT


def dataset_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:DATASET_HASH_LENGTH]


def row_count(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def dataset_settings(train_file: Path, val_file: Path) -> dict:
    return {
        "train_rows": row_count(train_file),
        "train_hash": dataset_hash(train_file),
        "val_rows": row_count(val_file),
        "val_hash": dataset_hash(val_file),
    }


def run_id_file(adapter_dir: Path) -> Path:
    return adapter_dir / RUN_ID_FILE_NAME


def write_run_id(adapter_dir: Path, run_id: str):
    run_id_file(adapter_dir).write_text(run_id, encoding="utf-8")


def read_run_id(adapter_dir: Path) -> str | None:
    if not run_id_file(adapter_dir).exists():
        return None
    return run_id_file(adapter_dir).read_text(encoding="utf-8").strip() or None


def eval_metrics(results: list[dict]) -> dict:
    return {**invariant_pass_rates(results), "val_rows": len(results)}


def eval_table(results: list[dict]) -> dict:
    return {column: [result[column] for result in results] for column in EVAL_TABLE_COLUMNS}


def mlflow_on_experiment():
    import mlflow

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def start_training_run(settings: dict, train_file: Path = TRAIN_FILE, val_file: Path = VAL_FILE) -> str | None:
    if not tracking_enabled():
        return None
    mlflow = mlflow_on_experiment()
    run = mlflow.start_run()
    mlflow.log_params({**settings, **dataset_settings(train_file, val_file), "git_commit": git_commit()})
    return run.info.run_id


def finish_training_run(run_id: str | None, adapter_dir: Path = ADAPTER_DIR):
    if run_id is None:
        return
    mlflow = mlflow_on_experiment()
    write_run_id(adapter_dir, run_id)
    mlflow.log_artifacts(str(adapter_dir), ADAPTER_ARTIFACT_PATH, run_id=run_id)
    mlflow.end_run()


def is_run_not_found(error) -> bool:
    return error.error_code in RUN_NOT_FOUND_ERROR_CODES


def run_id_known_to_server(mlflow, run_id: str | None) -> str | None:
    if run_id is None:
        return None
    try:
        mlflow.get_run(run_id)
    except mlflow.exceptions.MlflowException as error:
        if is_run_not_found(error):
            return None
        raise
    return run_id


def unknown_run_params(adapter_run_id: str | None, known_run_id: str | None) -> dict:
    if adapter_run_id == known_run_id:
        return {}
    return {UNKNOWN_RUN_ID_PARAM: adapter_run_id}


def remember_new_run(adapter_dir: Path, run_id: str):
    if adapter_dir.is_dir():
        write_run_id(adapter_dir, run_id)


def start_adapter_run(mlflow, adapter_dir: Path):
    adapter_run_id = read_run_id(adapter_dir)
    known_run_id = run_id_known_to_server(mlflow, adapter_run_id)
    run = mlflow.start_run(run_id=known_run_id)
    if known_run_id is None:
        remember_new_run(adapter_dir, run.info.run_id)
    mlflow.log_params(unknown_run_params(adapter_run_id, known_run_id))
    return run


def log_eval_run(results: list[dict], eval_file: Path = EVAL_FILE, adapter_dir: Path = ADAPTER_DIR) -> str | None:
    if not tracking_enabled():
        return None
    metrics = eval_metrics(results)
    mlflow = mlflow_on_experiment()
    with start_adapter_run(mlflow, adapter_dir) as run:
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(eval_file))
        mlflow.log_table(eval_table(results), EVAL_TABLE_FILE)
    return run.info.run_id


def baseline_metrics(results: list[dict]) -> dict:
    return {**invariant_pass_rates(results), "rows": len(results)}


def run_id_named(mlflow, run_name: str) -> str | None:
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        filter_string=f"attributes.run_name = '{run_name}'",
        output_format="list",
    )
    return runs[0].info.run_id if runs else None


def log_baseline_run(results: list[dict]) -> str | None:
    if not tracking_enabled():
        return None
    metrics = baseline_metrics(results)
    mlflow = mlflow_on_experiment()
    with mlflow.start_run(run_id=run_id_named(mlflow, BASELINE_RUN_NAME), run_name=BASELINE_RUN_NAME) as run:
        mlflow.log_metrics(metrics)
    return run.info.run_id


def rubric_hash(rubric: str) -> str:
    return hashlib.sha256(rubric.encode("utf-8")).hexdigest()[:RUBRIC_HASH_LENGTH]


def judge_metrics(summary: dict) -> dict:
    return {JUDGE_METRIC_PREFIX + name: value for name, value in summary.items()}


def judge_tags(judge_model: str, rubric: str) -> dict:
    return {"judge_model": judge_model, "judge_rubric_hash": rubric_hash(rubric)}


def table_cell(value):
    return LIST_CELL_SEPARATOR.join(value) if isinstance(value, list) else value


def table_of(rows: list[dict]) -> dict:
    return {column: [table_cell(row[column]) for row in rows] for column in rows[0]}


def log_judge_run(
    judged_rows: list[dict],
    summary: dict,
    judge_model: str,
    rubric: str,
    judge_file: Path = JUDGE_FILE,
    adapter_dir: Path = ADAPTER_DIR,
) -> str | None:
    if not tracking_enabled():
        return None
    if not judged_rows:
        raise ValueError("no judged rows to log")
    mlflow = mlflow_on_experiment()
    with start_adapter_run(mlflow, adapter_dir) as run:
        mlflow.set_tags(judge_tags(judge_model, rubric))
        mlflow.log_metrics(judge_metrics(summary))
        mlflow.log_artifact(str(judge_file))
        mlflow.log_table(table_of(judged_rows), JUDGE_TABLE_FILE)
    return run.info.run_id


def similarity_metrics(summary: dict) -> dict:
    return {SIMILARITY_METRIC_PREFIX + name: value for name, value in summary.items()}


def log_similarity_run(summary: dict, adapter_dir: Path = ADAPTER_DIR) -> str | None:
    if not tracking_enabled():
        return None
    if not summary.get("rows"):
        raise ValueError("no rows to log a similarity for")
    mlflow = mlflow_on_experiment()
    with start_adapter_run(mlflow, adapter_dir) as run:
        mlflow.log_metrics(similarity_metrics(summary))
    return run.info.run_id
