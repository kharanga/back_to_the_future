from types import SimpleNamespace

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from tag_analyzer import PROJECT_ROOT, tracking
from tag_analyzer.tracking import git_commit, log_eval_run, tracking_enabled

SETTINGS = {"K": 3, "THRESHOLD": 0.7, "clip_model": "fake-clip", "library_size": 170, "test_set_size": 2}
METRICS = {"brand_accuracy": 0.5, "era_accuracy": 1.0}
MISTAKES = ["b.jpg -> Champion/90s", "c.jpg -> Russell/80s"]


@pytest.fixture
def tracking_off(monkeypatch):
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", None)
    monkeypatch.setattr(mlflow, "start_run", lambda: pytest.fail("a run was started with tracking off"))


@pytest.fixture
def run_store(tmp_path, monkeypatch) -> MlflowClient:
    store_uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", store_uri)
    monkeypatch.setattr(tracking, "git_commit", lambda: "abc123")
    return MlflowClient(store_uri)


def test_tracking_enabled_is_true_when_the_tracking_uri_is_set(monkeypatch):
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", "http://fake.host:5000")
    assert tracking_enabled()


def test_tracking_enabled_is_false_when_the_tracking_uri_is_unset(monkeypatch):
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", None)
    assert not tracking_enabled()


def test_tracking_enabled_is_false_when_the_tracking_uri_is_empty(monkeypatch):
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", "")
    assert not tracking_enabled()


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


def test_log_eval_run_logs_nothing_when_the_tracking_uri_is_unset(tracking_off):
    assert log_eval_run(SETTINGS, METRICS, MISTAKES) is None


def test_log_eval_run_logs_nothing_when_the_tracking_uri_is_empty(tracking_off, monkeypatch):
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", "")
    assert log_eval_run(SETTINGS, METRICS, MISTAKES) is None


def test_log_eval_run_returns_the_id_of_one_new_run_in_the_tag_analyzer_experiment(run_store):
    run_id = log_eval_run(SETTINGS, METRICS, MISTAKES)
    experiment = run_store.get_experiment_by_name("tag-analyzer")
    assert [run.info.run_id for run in run_store.search_runs([experiment.experiment_id])] == [run_id]


def test_log_eval_run_logs_the_settings_and_the_git_commit_as_params(run_store):
    run_id = log_eval_run(SETTINGS, METRICS, MISTAKES)
    assert run_store.get_run(run_id).data.params == {
        "K": "3",
        "THRESHOLD": "0.7",
        "clip_model": "fake-clip",
        "library_size": "170",
        "test_set_size": "2",
        "git_commit": "abc123",
    }


def test_log_eval_run_logs_the_accuracies_as_metrics(run_store):
    run_id = log_eval_run(SETTINGS, METRICS, MISTAKES)
    assert run_store.get_run(run_id).data.metrics == {"brand_accuracy": 0.5, "era_accuracy": 1.0}


def test_log_eval_run_logs_the_mistakes_as_a_text_file_one_per_line(run_store, tmp_path):
    run_id = log_eval_run(SETTINGS, METRICS, MISTAKES)
    downloaded = run_store.download_artifacts(run_id, "mistakes.txt", str(tmp_path))
    assert open(downloaded, encoding="utf-8").read() == "b.jpg -> Champion/90s\nc.jpg -> Russell/80s"


def test_log_eval_run_starts_a_new_run_for_every_eval(run_store):
    assert log_eval_run(SETTINGS, METRICS, MISTAKES) != log_eval_run(SETTINGS, METRICS, MISTAKES)
