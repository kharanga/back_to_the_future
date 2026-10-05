import json
from types import SimpleNamespace

import mlflow
import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.tracking import MlflowClient

from listings import PROJECT_ROOT, config, tracking
from listings.tracking import (
    baseline_metrics,
    dataset_hash,
    dataset_settings,
    eval_metrics,
    eval_table,
    finish_training_run,
    git_commit,
    is_run_not_found,
    judge_metrics,
    judge_tags,
    log_baseline_run,
    log_eval_run,
    log_judge_run,
    log_similarity_run,
    mlflow_on_experiment,
    read_run_id,
    remember_new_run,
    report_to,
    row_count,
    rubric_hash,
    run_id_file,
    run_id_known_to_server,
    run_id_named,
    similarity_metrics,
    start_adapter_run,
    start_training_run,
    table_of,
    tracking_enabled,
    unknown_run_params,
    write_run_id,
)

SETTINGS = {"BASE_MODEL": "fake/base-model", "EPOCHS": 3}
EVERY_INVARIANT_PASSES = {
    "starts_with_diamond": True,
    "ends_with_period": True,
    "word_count_ok": True,
    "brand_mentioned": True,
    "era_mentioned": True,
}
RUSSELL_ROW = {
    "listing_id": 7,
    "target": "💎 Vintage 90's Boxy Russell Hoodie.",
    "generated": "💎 Vintage 90's Russell Hoodie.",
    **EVERY_INVARIANT_PASSES,
}
NIKE_ROW_WITHOUT_BRAND = {
    "listing_id": 8,
    "target": "💎 Vintage Y2K Nike Tee.",
    "generated": "💎 Vintage Y2K Tee.",
    **EVERY_INVARIANT_PASSES,
    "brand_mentioned": False,
}
RESULTS = [RUSSELL_ROW, NIKE_ROW_WITHOUT_BRAND]
RUN_ID_THE_SERVER_DOES_NOT_HAVE = "0123456789abcdef0123456789abcdef"
JUDGED_ROWS = [
    {
        "listing_id": 7,
        "target": "💎 Vintage 90's Boxy Russell Hoodie.",
        "generated": "💎 Vintage 90's Russell Hoodie.",
        "same_item": True,
        "brand_agrees": True,
        "era_agrees": True,
        "nothing_invented": True,
        "key_details_kept": True,
        "similarity": 95,
        "overall": "equivalent",
        "reason": "Same hoodie, brand and decade.",
    },
    {
        "listing_id": 8,
        "target": "💎 Vintage Y2K Nike Tee.",
        "generated": "💎 Vintage Y2K Nike Swoosh Tee.",
        "same_item": True,
        "brand_agrees": True,
        "era_agrees": True,
        "nothing_invented": False,
        "key_details_kept": True,
        "similarity": 30,
        "overall": "wrong",
        "reason": "The generated line adds a swoosh graphic.",
    },
]
JUDGE_SUMMARY = {
    "same_item": 1.0,
    "brand_agrees": 1.0,
    "era_agrees": 1.0,
    "nothing_invented": 0.5,
    "key_details_kept": 1.0,
    "similarity_mean": 62.5,
    "similarity_80_or_more": 0.5,
    "overall_equivalent": 0.5,
    "overall_acceptable": 0.0,
    "overall_wrong": 0.5,
    "rows": 2,
}
SIMILARITY_SUMMARY = {"exact_match": 0.25, "word_overlap_mean": 0.625, "word_overlap_median": 0.5, "rows": 4}


@pytest.fixture(autouse=True)
def no_run_left_active():
    yield
    mlflow.end_run()


@pytest.fixture
def tracking_off(monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", None)
    monkeypatch.setattr(mlflow, "start_run", lambda **options: pytest.fail("a run was started with tracking off"))


@pytest.fixture
def run_store(tmp_path, monkeypatch) -> MlflowClient:
    store_uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", store_uri)
    monkeypatch.setattr(tracking, "git_commit", lambda: "abc123")
    return MlflowClient(store_uri)


@pytest.fixture
def train_file(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"listing_id": 1}\n{"listing_id": 2}\n{"listing_id": 3}\n', encoding="utf-8")
    return path


@pytest.fixture
def val_file(tmp_path):
    path = tmp_path / "val.jsonl"
    path.write_text('{"listing_id": 4}\n', encoding="utf-8")
    return path


@pytest.fixture
def adapter_dir(tmp_path):
    path = tmp_path / "adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    return path


@pytest.fixture
def eval_file(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in RESULTS), encoding="utf-8")
    return path


@pytest.fixture
def judge_file(tmp_path):
    path = tmp_path / "judge.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in JUDGED_ROWS), encoding="utf-8")
    return path


@pytest.fixture
def training_run_id(run_store, train_file, val_file, adapter_dir) -> str:
    run_id = start_training_run(SETTINGS, train_file, val_file)
    finish_training_run(run_id, adapter_dir)
    return run_id


def artifact_paths(run_store: MlflowClient, run_id: str, folder: str | None = None) -> list[str]:
    return sorted(artifact.path for artifact in run_store.list_artifacts(run_id, folder))


def run_ids_in_the_experiment(run_store: MlflowClient) -> list[str]:
    experiment = run_store.get_experiment_by_name("listings-first-line")
    if experiment is None:
        return []
    return [run.info.run_id for run in run_store.search_runs([experiment.experiment_id])]


def test_tracking_enabled_is_true_when_the_tracking_uri_is_set(monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", "http://fake.host:5000")
    assert tracking_enabled()


def test_tracking_enabled_is_false_when_the_tracking_uri_is_unset(monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", None)
    assert not tracking_enabled()


def test_tracking_enabled_is_false_when_the_tracking_uri_is_empty(monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", "")
    assert not tracking_enabled()


def test_report_to_is_mlflow_when_tracking_is_on(monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", "http://fake.host:5000")
    assert report_to() == "mlflow"


def test_report_to_is_none_when_tracking_is_off(monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", None)
    assert report_to() == "none"


def test_git_commit_is_the_commit_hash_git_prints(monkeypatch):
    monkeypatch.setattr(tracking.subprocess, "run", lambda command, **options: SimpleNamespace(stdout="abc123\n"))
    assert git_commit() == "abc123"


def test_git_commit_asks_git_for_head_in_the_repo_root(monkeypatch):
    git_calls = []
    monkeypatch.setattr(
        tracking.subprocess,
        "run",
        lambda command, **options: git_calls.append((command, options["cwd"])) or SimpleNamespace(stdout="abc123\n"),
    )
    git_commit()
    assert git_calls == [(["git", "rev-parse", "HEAD"], PROJECT_ROOT)]


def test_git_commit_is_unknown_when_git_prints_nothing(monkeypatch):
    monkeypatch.setattr(tracking.subprocess, "run", lambda command, **options: SimpleNamespace(stdout=""))
    assert git_commit() == "unknown"


def test_git_commit_is_unknown_when_git_is_not_installed(monkeypatch):
    def git_is_missing(command, **options):
        raise FileNotFoundError("git")

    monkeypatch.setattr(tracking.subprocess, "run", git_is_missing)
    assert git_commit() == "unknown"


def test_dataset_hash_is_the_first_twelve_hex_characters_of_the_files_sha256(tmp_path):
    (tmp_path / "empty.jsonl").write_bytes(b"")
    assert dataset_hash(tmp_path / "empty.jsonl") == "e3b0c44298fc"


def test_dataset_hash_is_the_same_for_files_with_the_same_bytes(tmp_path):
    (tmp_path / "train.jsonl").write_bytes(b'{"listing_id": 1}\n')
    (tmp_path / "copy.jsonl").write_bytes(b'{"listing_id": 1}\n')
    assert dataset_hash(tmp_path / "train.jsonl") == dataset_hash(tmp_path / "copy.jsonl")


def test_dataset_hash_changes_when_one_byte_changes(tmp_path):
    (tmp_path / "train.jsonl").write_bytes(b'{"listing_id": 1}\n')
    (tmp_path / "changed.jsonl").write_bytes(b'{"listing_id": 2}\n')
    assert dataset_hash(tmp_path / "train.jsonl") != dataset_hash(tmp_path / "changed.jsonl")


def test_row_count_is_the_number_of_lines_in_the_file(train_file):
    assert row_count(train_file) == 3


def test_dataset_settings_are_the_row_count_and_hash_of_train_and_val(train_file, val_file):
    assert dataset_settings(train_file, val_file) == {
        "train_rows": 3,
        "train_hash": dataset_hash(train_file),
        "val_rows": 1,
        "val_hash": dataset_hash(val_file),
    }


def test_run_id_file_is_mlflow_run_id_txt_inside_the_adapter_folder(adapter_dir):
    assert run_id_file(adapter_dir) == adapter_dir / "mlflow_run_id.txt"


def test_write_run_id_writes_the_run_id_into_the_adapter_folder(adapter_dir):
    write_run_id(adapter_dir, "run-abc")
    assert (adapter_dir / "mlflow_run_id.txt").read_text(encoding="utf-8") == "run-abc"


def test_read_run_id_reads_back_what_write_run_id_wrote(adapter_dir):
    write_run_id(adapter_dir, "run-abc")
    assert read_run_id(adapter_dir) == "run-abc"


def test_read_run_id_is_none_when_the_adapter_folder_has_no_run_id_file(adapter_dir):
    assert read_run_id(adapter_dir) is None


def test_read_run_id_is_none_when_the_run_id_file_is_empty(adapter_dir):
    write_run_id(adapter_dir, "")
    assert read_run_id(adapter_dir) is None


def test_read_run_id_ignores_a_trailing_newline(adapter_dir):
    write_run_id(adapter_dir, "run-abc\n")
    assert read_run_id(adapter_dir) == "run-abc"


def test_eval_metrics_are_the_five_pass_rates_and_the_val_row_count():
    assert eval_metrics(RESULTS) == {
        "starts_with_diamond": 1.0,
        "ends_with_period": 1.0,
        "word_count_ok": 1.0,
        "brand_mentioned": 0.5,
        "era_mentioned": 1.0,
        "val_rows": 2,
    }


def test_eval_metrics_raises_when_there_are_no_rows_to_score():
    with pytest.raises(ZeroDivisionError):
        eval_metrics([])


def test_eval_table_lists_each_column_across_the_rows():
    assert eval_table(RESULTS) == {
        "listing_id": [7, 8],
        "target": ["💎 Vintage 90's Boxy Russell Hoodie.", "💎 Vintage Y2K Nike Tee."],
        "generated": ["💎 Vintage 90's Russell Hoodie.", "💎 Vintage Y2K Tee."],
        "starts_with_diamond": [True, True],
        "ends_with_period": [True, True],
        "word_count_ok": [True, True],
        "brand_mentioned": [True, False],
        "era_mentioned": [True, True],
    }


def test_mlflow_on_experiment_points_mlflow_at_the_configured_tracking_uri(run_store):
    assert mlflow_on_experiment().get_tracking_uri() == config.MLFLOW_TRACKING_URI


def test_mlflow_on_experiment_creates_the_listings_first_line_experiment(run_store):
    mlflow_on_experiment()
    assert run_store.get_experiment_by_name("listings-first-line") is not None


def test_start_training_run_does_nothing_when_tracking_is_off(tracking_off, train_file, val_file):
    assert start_training_run(SETTINGS, train_file, val_file) is None


def test_start_training_run_opens_one_run_in_the_listings_first_line_experiment(run_store, train_file, val_file):
    run_id = start_training_run(SETTINGS, train_file, val_file)
    assert run_ids_in_the_experiment(run_store) == [run_id]


def test_start_training_run_logs_the_settings_the_datasets_and_the_git_commit(run_store, train_file, val_file):
    run_id = start_training_run(SETTINGS, train_file, val_file)
    assert run_store.get_run(run_id).data.params == {
        "BASE_MODEL": "fake/base-model",
        "EPOCHS": "3",
        "train_rows": "3",
        "train_hash": dataset_hash(train_file),
        "val_rows": "1",
        "val_hash": dataset_hash(val_file),
        "git_commit": "abc123",
    }


def test_start_training_run_leaves_the_run_active_for_the_trainer_to_log_into(run_store, train_file, val_file):
    run_id = start_training_run(SETTINGS, train_file, val_file)
    assert mlflow.active_run().info.run_id == run_id


def test_finish_training_run_does_nothing_when_there_is_no_run(tracking_off, adapter_dir):
    finish_training_run(None, adapter_dir)
    assert not (adapter_dir / "mlflow_run_id.txt").exists()


def test_finish_training_run_writes_the_run_id_into_the_adapter_folder(training_run_id, adapter_dir):
    assert read_run_id(adapter_dir) == training_run_id


def test_finish_training_run_uploads_the_whole_adapter_folder_to_the_run(run_store, training_run_id):
    assert artifact_paths(run_store, training_run_id, "adapter") == [
        "adapter/adapter_config.json",
        "adapter/mlflow_run_id.txt",
    ]


def test_finish_training_run_ends_the_run(run_store, training_run_id):
    assert mlflow.active_run() is None
    assert run_store.get_run(training_run_id).info.status == "FINISHED"


def test_is_run_not_found_is_true_when_the_server_says_the_run_does_not_exist():
    assert is_run_not_found(SimpleNamespace(error_code="RESOURCE_DOES_NOT_EXIST"))


def test_is_run_not_found_is_true_when_the_server_rejects_the_run_id_as_invalid():
    assert is_run_not_found(SimpleNamespace(error_code="INVALID_PARAMETER_VALUE"))


def test_is_run_not_found_is_false_for_any_other_server_error():
    assert not is_run_not_found(SimpleNamespace(error_code="INTERNAL_ERROR"))


def test_run_id_known_to_server_is_the_run_id_when_the_server_has_the_run(training_run_id):
    assert run_id_known_to_server(mlflow_on_experiment(), training_run_id) == training_run_id


def test_run_id_known_to_server_is_none_when_the_server_does_not_have_the_run(run_store):
    assert run_id_known_to_server(mlflow_on_experiment(), RUN_ID_THE_SERVER_DOES_NOT_HAVE) is None


def test_run_id_known_to_server_is_none_when_there_is_no_run_id(run_store):
    assert run_id_known_to_server(mlflow_on_experiment(), None) is None


def test_run_id_known_to_server_raises_any_other_server_error():
    def server_is_broken(run_id):
        raise MlflowException("fake server error", error_code=INTERNAL_ERROR)

    broken_mlflow = SimpleNamespace(get_run=server_is_broken, exceptions=mlflow.exceptions)
    with pytest.raises(MlflowException):
        run_id_known_to_server(broken_mlflow, "run-abc")


def test_unknown_run_params_are_empty_when_the_server_knows_the_adapters_run():
    assert unknown_run_params("run-abc", "run-abc") == {}


def test_unknown_run_params_are_empty_when_the_adapter_has_no_run_id():
    assert unknown_run_params(None, None) == {}


def test_unknown_run_params_name_the_run_id_the_server_does_not_have():
    assert unknown_run_params("run-abc", None) == {"adapter_run_id_not_found": "run-abc"}


def test_remember_new_run_writes_the_run_id_into_an_existing_adapter_folder(adapter_dir):
    remember_new_run(adapter_dir, "run-abc")
    assert read_run_id(adapter_dir) == "run-abc"


def test_remember_new_run_creates_nothing_when_the_adapter_folder_does_not_exist(tmp_path):
    remember_new_run(tmp_path / "no_adapter_here", "run-abc")
    assert not (tmp_path / "no_adapter_here").exists()


def test_start_adapter_run_reopens_the_run_named_in_the_adapter_folder(training_run_id, adapter_dir):
    assert start_adapter_run(mlflow_on_experiment(), adapter_dir).info.run_id == training_run_id


def test_start_adapter_run_starts_a_new_run_when_the_adapter_folder_has_no_run_id(run_store, adapter_dir):
    run = start_adapter_run(mlflow_on_experiment(), adapter_dir)
    assert run_ids_in_the_experiment(run_store) == [run.info.run_id]


def test_start_adapter_run_records_a_run_id_the_server_does_not_have_on_a_new_run(run_store, adapter_dir):
    write_run_id(adapter_dir, RUN_ID_THE_SERVER_DOES_NOT_HAVE)
    run = start_adapter_run(mlflow_on_experiment(), adapter_dir)
    assert run_store.get_run(run.info.run_id).data.params == {
        "adapter_run_id_not_found": RUN_ID_THE_SERVER_DOES_NOT_HAVE
    }


def test_start_adapter_run_remembers_a_new_run_in_the_adapter_folder(run_store, adapter_dir):
    run = start_adapter_run(mlflow_on_experiment(), adapter_dir)
    assert read_run_id(adapter_dir) == run.info.run_id


def test_start_adapter_run_replaces_a_run_id_the_server_does_not_have_with_the_new_runs_id(run_store, adapter_dir):
    write_run_id(adapter_dir, RUN_ID_THE_SERVER_DOES_NOT_HAVE)
    run = start_adapter_run(mlflow_on_experiment(), adapter_dir)
    assert read_run_id(adapter_dir) == run.info.run_id


def test_start_adapter_run_does_not_rewrite_the_file_of_a_run_the_server_has(training_run_id, adapter_dir):
    written_at = run_id_file(adapter_dir).stat().st_mtime_ns
    start_adapter_run(mlflow_on_experiment(), adapter_dir)
    assert run_id_file(adapter_dir).stat().st_mtime_ns == written_at
    assert read_run_id(adapter_dir) == training_run_id


def test_start_adapter_run_starts_a_new_run_and_writes_nothing_when_the_adapter_folder_is_missing(
    run_store, tmp_path
):
    run = start_adapter_run(mlflow_on_experiment(), tmp_path / "no_adapter_here")
    assert run_ids_in_the_experiment(run_store) == [run.info.run_id]
    assert not (tmp_path / "no_adapter_here").exists()


def test_start_adapter_run_leaves_the_run_open_for_the_caller_to_log_into(run_store, adapter_dir):
    run = start_adapter_run(mlflow_on_experiment(), adapter_dir)
    assert mlflow.active_run().info.run_id == run.info.run_id


def test_log_eval_run_does_nothing_when_tracking_is_off(tracking_off, eval_file, adapter_dir):
    assert log_eval_run(RESULTS, eval_file, adapter_dir) is None


def test_log_eval_run_adds_to_the_training_run_named_in_the_adapter_folder(training_run_id, eval_file, adapter_dir):
    assert log_eval_run(RESULTS, eval_file, adapter_dir) == training_run_id


def test_log_eval_run_logs_the_pass_rates_and_the_val_row_count_as_metrics(run_store, eval_file, adapter_dir):
    run_id = log_eval_run(RESULTS, eval_file, adapter_dir)
    assert run_store.get_run(run_id).data.metrics == {
        "starts_with_diamond": 1.0,
        "ends_with_period": 1.0,
        "word_count_ok": 1.0,
        "brand_mentioned": 0.5,
        "era_mentioned": 1.0,
        "val_rows": 2.0,
    }


def test_log_eval_run_uploads_eval_jsonl_and_the_rows_table(run_store, eval_file, adapter_dir):
    run_id = log_eval_run(RESULTS, eval_file, adapter_dir)
    assert artifact_paths(run_store, run_id) == ["eval.jsonl", "eval_rows.json"]


def test_log_eval_run_logs_the_rows_as_a_table_of_lines_and_flags(run_store, eval_file, adapter_dir):
    run_id = log_eval_run(RESULTS, eval_file, adapter_dir)
    logged_table = mlflow.load_table("eval_rows.json", run_ids=[run_id])
    assert logged_table.to_dict("list") == eval_table(RESULTS)


def test_log_eval_run_keeps_the_training_settings_on_the_shared_run(run_store, training_run_id, eval_file, adapter_dir):
    log_eval_run(RESULTS, eval_file, adapter_dir)
    assert run_store.get_run(training_run_id).data.params["BASE_MODEL"] == "fake/base-model"


def test_log_eval_run_starts_a_new_run_when_the_adapter_folder_has_no_run_id(
    run_store, training_run_id, eval_file, tmp_path
):
    adapter_without_run_id = tmp_path / "adapter_copied_without_run_id"
    adapter_without_run_id.mkdir()
    new_run_id = log_eval_run(RESULTS, eval_file, adapter_without_run_id)
    assert new_run_id not in (None, training_run_id)


def test_log_eval_run_reuses_the_run_it_started_for_an_adapter_without_a_run_id(run_store, eval_file, adapter_dir):
    first_run_id = log_eval_run(RESULTS, eval_file, adapter_dir)
    second_run_id = log_eval_run(RESULTS, eval_file, adapter_dir)
    assert run_ids_in_the_experiment(run_store) == [first_run_id] == [second_run_id]


def test_log_eval_run_logs_no_params_on_a_new_run_for_an_adapter_without_a_run_id(run_store, eval_file, adapter_dir):
    run_id = log_eval_run(RESULTS, eval_file, adapter_dir)
    assert run_store.get_run(run_id).data.params == {}


def test_log_eval_run_starts_a_new_run_when_the_server_does_not_have_the_adapters_run(
    run_store, training_run_id, eval_file, adapter_dir
):
    write_run_id(adapter_dir, RUN_ID_THE_SERVER_DOES_NOT_HAVE)
    new_run_id = log_eval_run(RESULTS, eval_file, adapter_dir)
    assert new_run_id not in (None, training_run_id, RUN_ID_THE_SERVER_DOES_NOT_HAVE)


def test_log_eval_run_records_the_unknown_run_id_on_the_new_run(run_store, eval_file, adapter_dir):
    write_run_id(adapter_dir, RUN_ID_THE_SERVER_DOES_NOT_HAVE)
    run_id = log_eval_run(RESULTS, eval_file, adapter_dir)
    assert run_store.get_run(run_id).data.params == {"adapter_run_id_not_found": RUN_ID_THE_SERVER_DOES_NOT_HAVE}


def test_log_eval_run_logs_the_scores_on_the_new_run_for_an_unknown_run_id(run_store, eval_file, adapter_dir):
    write_run_id(adapter_dir, RUN_ID_THE_SERVER_DOES_NOT_HAVE)
    run_id = log_eval_run(RESULTS, eval_file, adapter_dir)
    assert run_store.get_run(run_id).data.metrics["brand_mentioned"] == 0.5


def test_log_eval_run_records_no_unknown_run_id_when_the_server_has_the_adapters_run(
    run_store, training_run_id, eval_file, adapter_dir
):
    log_eval_run(RESULTS, eval_file, adapter_dir)
    assert "adapter_run_id_not_found" not in run_store.get_run(training_run_id).data.params


def test_log_eval_run_raises_for_no_rows_and_leaves_the_training_run_finished(
    run_store, training_run_id, eval_file, adapter_dir
):
    with pytest.raises(ZeroDivisionError):
        log_eval_run([], eval_file, adapter_dir)
    assert run_store.get_run(training_run_id).info.status == "FINISHED"


def test_log_eval_run_raises_for_no_rows_and_leaves_no_empty_run_behind(run_store, eval_file, adapter_dir):
    with pytest.raises(ZeroDivisionError):
        log_eval_run([], eval_file, adapter_dir)
    assert run_ids_in_the_experiment(run_store) == []


def test_baseline_metrics_are_the_five_pass_rates_and_the_row_count():
    assert baseline_metrics(RESULTS) == {
        "starts_with_diamond": 1.0,
        "ends_with_period": 1.0,
        "word_count_ok": 1.0,
        "brand_mentioned": 0.5,
        "era_mentioned": 1.0,
        "rows": 2,
    }


def test_baseline_metrics_raises_when_there_are_no_rows_to_score():
    with pytest.raises(ZeroDivisionError):
        baseline_metrics([])


def test_run_id_named_is_the_id_of_the_run_with_that_name(run_store):
    baseline_run_id = log_baseline_run(RESULTS)
    assert run_id_named(mlflow_on_experiment(), "shop-baseline") == baseline_run_id


def test_run_id_named_is_none_when_no_run_has_that_name(run_store):
    assert run_id_named(mlflow_on_experiment(), "shop-baseline") is None


def test_run_id_named_ignores_runs_with_other_names(run_store, training_run_id):
    assert run_id_named(mlflow_on_experiment(), "shop-baseline") is None


def test_log_baseline_run_does_nothing_when_tracking_is_off(tracking_off):
    assert log_baseline_run(RESULTS) is None


def test_log_baseline_run_logs_to_a_run_named_shop_baseline(run_store):
    run_id = log_baseline_run(RESULTS)
    assert run_store.get_run(run_id).info.run_name == "shop-baseline"


def test_log_baseline_run_puts_the_run_in_the_listings_first_line_experiment(run_store):
    run_id = log_baseline_run(RESULTS)
    assert run_ids_in_the_experiment(run_store) == [run_id]


def test_log_baseline_run_logs_the_five_pass_rates_and_the_row_count(run_store):
    run_id = log_baseline_run(RESULTS)
    assert run_store.get_run(run_id).data.metrics == {
        "starts_with_diamond": 1.0,
        "ends_with_period": 1.0,
        "word_count_ok": 1.0,
        "brand_mentioned": 0.5,
        "era_mentioned": 1.0,
        "rows": 2.0,
    }


def test_log_baseline_run_reuses_the_same_run_when_run_twice(run_store):
    first_run_id = log_baseline_run(RESULTS)
    second_run_id = log_baseline_run(RESULTS)
    assert run_ids_in_the_experiment(run_store) == [first_run_id] == [second_run_id]


def test_log_baseline_run_keeps_the_latest_scores_when_run_twice(run_store):
    log_baseline_run(RESULTS)
    run_id = log_baseline_run([RUSSELL_ROW])
    assert run_store.get_run(run_id).data.metrics["brand_mentioned"] == 1.0


def test_log_baseline_run_leaves_a_training_run_alone(run_store, training_run_id):
    baseline_run_id = log_baseline_run(RESULTS)
    assert baseline_run_id != training_run_id
    assert run_store.get_run(training_run_id).data.metrics == {}


def test_log_baseline_run_raises_for_no_rows_and_leaves_no_run_behind(run_store):
    with pytest.raises(ZeroDivisionError):
        log_baseline_run([])
    assert run_ids_in_the_experiment(run_store) == []


def test_rubric_hash_is_the_first_twelve_hex_characters_of_the_texts_sha256():
    assert rubric_hash("") == "e3b0c44298fc"


def test_rubric_hash_is_the_same_for_the_same_rubric():
    assert rubric_hash("Compare the two lines.") == rubric_hash("Compare the two lines.")


def test_rubric_hash_changes_when_the_rubric_changes():
    assert rubric_hash("Compare the two lines.") != rubric_hash("Compare the two lines carefully.")


def test_judge_metrics_are_the_summary_with_judge_in_front_of_each_name():
    assert judge_metrics({"same_item": 1.0, "overall_wrong": 0.5, "rows": 2}) == {
        "judge_same_item": 1.0,
        "judge_overall_wrong": 0.5,
        "judge_rows": 2,
    }


def test_judge_tags_are_the_judge_model_and_the_rubric_hash():
    assert judge_tags("claude-haiku-4-5", "") == {
        "judge_model": "claude-haiku-4-5",
        "judge_rubric_hash": "e3b0c44298fc",
    }


def test_table_of_lists_each_column_of_the_first_row_across_the_rows():
    assert table_of([{"listing_id": 7, "overall": "equivalent"}, {"listing_id": 8, "overall": "wrong"}]) == {
        "listing_id": [7, 8],
        "overall": ["equivalent", "wrong"],
    }






def test_log_judge_run_does_nothing_when_tracking_is_off(tracking_off, judge_file, adapter_dir):
    assert log_judge_run(JUDGED_ROWS, JUDGE_SUMMARY, "claude-haiku-4-5", "rubric", judge_file, adapter_dir) is None


def test_log_judge_run_adds_to_the_training_run_named_in_the_adapter_folder(training_run_id, judge_file, adapter_dir):
    run_id = log_judge_run(JUDGED_ROWS, JUDGE_SUMMARY, "claude-haiku-4-5", "rubric", judge_file, adapter_dir)
    assert run_id == training_run_id


def test_log_judge_run_logs_every_summary_value_as_a_judge_metric(run_store, judge_file, adapter_dir):
    run_id = log_judge_run(JUDGED_ROWS, JUDGE_SUMMARY, "claude-haiku-4-5", "rubric", judge_file, adapter_dir)
    assert run_store.get_run(run_id).data.metrics == {
        "judge_same_item": 1.0,
        "judge_brand_agrees": 1.0,
        "judge_era_agrees": 1.0,
        "judge_nothing_invented": 0.5,
        "judge_key_details_kept": 1.0,
        "judge_similarity_mean": 62.5,
        "judge_similarity_80_or_more": 0.5,
        "judge_overall_equivalent": 0.5,
        "judge_overall_acceptable": 0.0,
        "judge_overall_wrong": 0.5,
        "judge_rows": 2.0,
    }



def test_log_judge_run_tags_the_run_with_the_judge_model_and_the_rubric_hash(run_store, judge_file, adapter_dir):
    run_id = log_judge_run(JUDGED_ROWS, JUDGE_SUMMARY, "claude-haiku-4-5", "rubric", judge_file, adapter_dir)
    tags = run_store.get_run(run_id).data.tags
    assert (tags["judge_model"], tags["judge_rubric_hash"]) == ("claude-haiku-4-5", rubric_hash("rubric"))


def test_log_judge_run_logs_the_judge_model_and_rubric_as_tags_not_params(run_store, judge_file, adapter_dir):
    run_id = log_judge_run(JUDGED_ROWS, JUDGE_SUMMARY, "claude-haiku-4-5", "rubric", judge_file, adapter_dir)
    assert run_store.get_run(run_id).data.params == {}


def test_log_judge_run_uploads_judge_jsonl_and_the_rows_table(run_store, judge_file, adapter_dir):
    run_id = log_judge_run(JUDGED_ROWS, JUDGE_SUMMARY, "claude-haiku-4-5", "rubric", judge_file, adapter_dir)
    assert artifact_paths(run_store, run_id) == ["judge.jsonl", "judge_rows.json"]


def test_log_judge_run_logs_the_judged_rows_as_a_table(run_store, judge_file, adapter_dir):
    run_id = log_judge_run(JUDGED_ROWS, JUDGE_SUMMARY, "claude-haiku-4-5", "rubric", judge_file, adapter_dir)
    logged_table = mlflow.load_table("judge_rows.json", run_ids=[run_id])
    assert logged_table.to_dict("list") == table_of(JUDGED_ROWS)


def test_log_judge_run_records_a_run_id_the_server_does_not_have_on_a_new_run(run_store, judge_file, adapter_dir):
    write_run_id(adapter_dir, RUN_ID_THE_SERVER_DOES_NOT_HAVE)
    run_id = log_judge_run(JUDGED_ROWS, JUDGE_SUMMARY, "claude-haiku-4-5", "rubric", judge_file, adapter_dir)
    assert run_store.get_run(run_id).data.params["adapter_run_id_not_found"] == RUN_ID_THE_SERVER_DOES_NOT_HAVE


def test_log_judge_run_raises_for_no_judged_rows_and_leaves_the_training_run_finished(
    run_store, training_run_id, judge_file, adapter_dir
):
    with pytest.raises(ValueError):
        log_judge_run([], JUDGE_SUMMARY, "claude-haiku-4-5", "rubric", judge_file, adapter_dir)
    assert run_store.get_run(training_run_id).info.status == "FINISHED"


def test_log_judge_run_raises_for_no_judged_rows_and_leaves_no_empty_run_behind(run_store, judge_file, adapter_dir):
    with pytest.raises(ValueError):
        log_judge_run([], JUDGE_SUMMARY, "claude-haiku-4-5", "rubric", judge_file, adapter_dir)
    assert run_ids_in_the_experiment(run_store) == []


def test_log_judge_run_logs_again_to_the_same_run_when_rejudged_with_the_same_rubric(
    run_store, training_run_id, judge_file, adapter_dir
):
    log_judge_run(JUDGED_ROWS, JUDGE_SUMMARY, "claude-haiku-4-5", "rubric", judge_file, adapter_dir)
    log_judge_run(JUDGED_ROWS, JUDGE_SUMMARY, "claude-haiku-4-5", "rubric", judge_file, adapter_dir)
    assert run_ids_in_the_experiment(run_store) == [training_run_id]


def test_log_judge_run_logs_to_the_same_run_when_rejudged_with_a_different_rubric_and_model(
    run_store, training_run_id, judge_file, adapter_dir
):
    log_judge_run(JUDGED_ROWS, JUDGE_SUMMARY, "claude-haiku-4-5", "rubric", judge_file, adapter_dir)
    run_id = log_judge_run(JUDGED_ROWS, JUDGE_SUMMARY, "fake-newer-model", "changed rubric", judge_file, adapter_dir)
    assert run_id == training_run_id
    assert run_ids_in_the_experiment(run_store) == [training_run_id]


def test_log_judge_run_tags_show_the_latest_model_and_rubric_after_rejudging(
    run_store, training_run_id, judge_file, adapter_dir
):
    log_judge_run(JUDGED_ROWS, JUDGE_SUMMARY, "claude-haiku-4-5", "rubric", judge_file, adapter_dir)
    log_judge_run(JUDGED_ROWS, JUDGE_SUMMARY, "fake-newer-model", "changed rubric", judge_file, adapter_dir)
    tags = run_store.get_run(training_run_id).data.tags
    assert (tags["judge_model"], tags["judge_rubric_hash"]) == ("fake-newer-model", rubric_hash("changed rubric"))


def test_log_judge_run_leaves_the_training_run_finished_after_rejudging(
    run_store, training_run_id, judge_file, adapter_dir
):
    log_judge_run(JUDGED_ROWS, JUDGE_SUMMARY, "claude-haiku-4-5", "rubric", judge_file, adapter_dir)
    log_judge_run(JUDGED_ROWS, JUDGE_SUMMARY, "fake-newer-model", "changed rubric", judge_file, adapter_dir)
    assert run_store.get_run(training_run_id).info.status == "FINISHED"


def test_similarity_metrics_are_the_summary_with_similarity_in_front_of_each_name():
    assert similarity_metrics(SIMILARITY_SUMMARY) == {
        "similarity_exact_match": 0.25,
        "similarity_word_overlap_mean": 0.625,
        "similarity_word_overlap_median": 0.5,
        "similarity_rows": 4,
    }


def test_log_similarity_run_does_nothing_when_tracking_is_off(tracking_off, adapter_dir):
    assert log_similarity_run(SIMILARITY_SUMMARY, adapter_dir) is None


def test_log_similarity_run_adds_to_the_training_run_named_in_the_adapter_folder(training_run_id, adapter_dir):
    assert log_similarity_run(SIMILARITY_SUMMARY, adapter_dir) == training_run_id


def test_log_similarity_run_logs_the_four_similarity_metrics(run_store, adapter_dir):
    run_id = log_similarity_run(SIMILARITY_SUMMARY, adapter_dir)
    assert run_store.get_run(run_id).data.metrics == {
        "similarity_exact_match": 0.25,
        "similarity_word_overlap_mean": 0.625,
        "similarity_word_overlap_median": 0.5,
        "similarity_rows": 4.0,
    }


def test_log_similarity_run_leaves_the_training_run_finished(run_store, training_run_id, adapter_dir):
    log_similarity_run(SIMILARITY_SUMMARY, adapter_dir)
    assert run_store.get_run(training_run_id).info.status == "FINISHED"


def test_log_similarity_run_starts_a_new_run_when_the_adapter_folder_has_no_run_id(run_store, adapter_dir):
    run_id = log_similarity_run(SIMILARITY_SUMMARY, adapter_dir)
    assert run_ids_in_the_experiment(run_store) == [run_id]


def test_log_similarity_run_raises_for_no_rows_and_leaves_the_training_run_finished(
    run_store, training_run_id, adapter_dir
):
    with pytest.raises(ValueError):
        log_similarity_run({**SIMILARITY_SUMMARY, "rows": 0}, adapter_dir)
    assert run_store.get_run(training_run_id).info.status == "FINISHED"


def test_log_similarity_run_raises_for_no_rows_and_leaves_no_empty_run_behind(run_store, adapter_dir):
    with pytest.raises(ValueError):
        log_similarity_run({**SIMILARITY_SUMMARY, "rows": 0}, adapter_dir)
    assert run_ids_in_the_experiment(run_store) == []


def test_log_similarity_run_shares_one_run_with_the_judge_for_an_adapter_without_a_run_id(
    run_store, judge_file, adapter_dir
):
    similarity_run_id = log_similarity_run(SIMILARITY_SUMMARY, adapter_dir)
    judge_run_id = log_judge_run(JUDGED_ROWS, JUDGE_SUMMARY, "claude-haiku-4-5", "rubric", judge_file, adapter_dir)
    assert run_ids_in_the_experiment(run_store) == [similarity_run_id] == [judge_run_id]


def test_log_similarity_run_and_the_judge_leave_both_sets_of_metrics_on_the_shared_run(
    run_store, judge_file, adapter_dir
):
    run_id = log_similarity_run(SIMILARITY_SUMMARY, adapter_dir)
    log_judge_run(JUDGED_ROWS, JUDGE_SUMMARY, "claude-haiku-4-5", "rubric", judge_file, adapter_dir)
    metrics = run_store.get_run(run_id).data.metrics
    assert (metrics["similarity_rows"], metrics["judge_rows"]) == (4.0, 2.0)


def test_log_similarity_run_logs_to_a_new_run_when_the_adapter_folder_is_missing(run_store, tmp_path):
    run_id = log_similarity_run(SIMILARITY_SUMMARY, tmp_path / "no_adapter_here")
    assert run_ids_in_the_experiment(run_store) == [run_id]
    assert not (tmp_path / "no_adapter_here").exists()
