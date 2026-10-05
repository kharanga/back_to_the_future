import json
from contextlib import nullcontext

import pytest

from tag_analyzer import eval as tag_eval
from tag_analyzer import store, tracking
from tag_analyzer.eval import accuracy, accuracy_lines, neighbors_for_each_photo, run_metrics, run_settings, score
from tag_analyzer.match import THRESHOLD
from tag_analyzer.schema import EvalScore, Neighbor

RUSSELL_90S_LABELS = [
    {"filename": "a.jpg", "brand": "Russell", "era": "90s"},
    {"filename": "b.jpg", "brand": "Russell", "era": "90s"},
]
ONE_OF_TWO_BRANDS_RIGHT = EvalScore(test_set_size=2, correct_brand=1, correct_era=2, mistakes=["b.jpg -> Champion/90s"])
EVERYTHING_RIGHT = EvalScore(test_set_size=2, correct_brand=2, correct_era=2, mistakes=[])


def neighbor(brand: str, era: str, similarity: float = 0.9) -> Neighbor:
    return Neighbor(filename=f"{brand}/{era}.jpg", brand=brand, era=era, similarity=similarity)


@pytest.fixture(autouse=True)
def tracking_off(monkeypatch):
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", None)


@pytest.fixture
def run_eval(tmp_path, monkeypatch, capsys):
    def printed_report(test_labels: list[dict], neighbors_by_filename: dict) -> str:
        (tmp_path / "test_labels.json").write_text(json.dumps(test_labels))
        monkeypatch.setattr(tag_eval, "TEST_LABELS_FILE", tmp_path / "test_labels.json")
        monkeypatch.setattr(tag_eval, "QUERIES_DIR", tmp_path)
        monkeypatch.setattr(store, "connect_from_env", lambda: nullcontext("fake client"))
        monkeypatch.setattr(store, "library_size", lambda client: 142)
        monkeypatch.setattr(
            store,
            "search_nearest_photos",
            lambda client, query_photo, k: neighbors_by_filename[query_photo.name],
        )
        tag_eval.main()
        return capsys.readouterr().out

    return printed_report


def test_neighbors_for_each_photo_searches_the_queries_folder_for_every_label(tmp_path, monkeypatch):
    monkeypatch.setattr(tag_eval, "QUERIES_DIR", tmp_path)
    monkeypatch.setattr(
        store, "search_nearest_photos", lambda client, query_photo, k: [f"{k} nearest to {query_photo}"]
    )
    assert neighbors_for_each_photo("fake client", RUSSELL_90S_LABELS) == {
        "a.jpg": [f"{tag_eval.K} nearest to {tmp_path / 'a.jpg'}"],
        "b.jpg": [f"{tag_eval.K} nearest to {tmp_path / 'b.jpg'}"],
    }


def test_score_counts_correct_brands_and_eras_and_lists_the_mistakes():
    neighbors_by_filename = {"a.jpg": [neighbor("Russell", "90s")], "b.jpg": [neighbor("Champion", "90s")]}
    assert score(RUSSELL_90S_LABELS, neighbors_by_filename) == ONE_OF_TWO_BRANDS_RIGHT


def test_score_has_no_mistakes_when_every_prediction_is_right():
    neighbors_by_filename = {"a.jpg": [neighbor("Russell", "90s")], "b.jpg": [neighbor("Russell", "90s")]}
    assert score(RUSSELL_90S_LABELS, neighbors_by_filename) == EVERYTHING_RIGHT


def test_score_lists_a_wrong_era_as_a_mistake_even_when_the_brand_is_right():
    labels = [{"filename": "a.jpg", "brand": "Russell", "era": "90s"}]
    assert score(labels, {"a.jpg": [neighbor("Russell", "80s")]}).mistakes == ["a.jpg -> Russell/80s"]


def test_score_compares_brand_and_era_ignoring_case():
    labels = [{"filename": "a.jpg", "brand": "RUSSELL", "era": "90S"}]
    assert score(labels, {"a.jpg": [neighbor("Russell", "90s")]}).mistakes == []


def test_score_counts_a_prediction_below_the_in_library_threshold():
    labels = [{"filename": "a.jpg", "brand": "Russell", "era": "90s"}]
    neighbors_by_filename = {"a.jpg": [neighbor("Russell", "90s", similarity=THRESHOLD - 0.1)]}
    assert score(labels, neighbors_by_filename).correct_brand == 1


def test_accuracy_is_the_correct_count_as_a_fraction_of_the_total():
    assert accuracy(16, 20) == 0.8


def test_accuracy_lines_are_the_brand_line_then_the_era_line():
    assert accuracy_lines(EVERYTHING_RIGHT) == [
        "brand accuracy: 2/2 = 100.0%",
        "era accuracy:   2/2 = 100.0%",
    ]


def test_accuracy_lines_end_with_the_mistakes_when_there_are_any():
    assert accuracy_lines(ONE_OF_TWO_BRANDS_RIGHT) == [
        "brand accuracy: 1/2 = 50.0%",
        "era accuracy:   2/2 = 100.0%",
        "\nmistakes:",
        "  b.jpg -> Champion/90s",
    ]


def test_run_settings_are_k_threshold_clip_model_library_size_and_test_set_size():
    assert run_settings(library_size=170, test_set_size=19) == {
        "K": tag_eval.K,
        "THRESHOLD": THRESHOLD,
        "clip_model": "clip-ViT-B-32-multilingual-v1",
        "library_size": 170,
        "test_set_size": 19,
    }


def test_run_metrics_are_brand_and_era_accuracy_as_fractions():
    assert run_metrics(ONE_OF_TWO_BRANDS_RIGHT) == {"brand_accuracy": 0.5, "era_accuracy": 1.0}


def test_main_prints_the_library_size_and_the_test_set_size(run_eval):
    report = run_eval(RUSSELL_90S_LABELS, {"a.jpg": [neighbor("Russell", "90s")], "b.jpg": [neighbor("Russell", "90s")]})
    assert report.splitlines()[0] == "library: 142 photos | test set: 2"


def test_main_reports_brand_accuracy_over_the_test_labels(run_eval):
    report = run_eval(
        RUSSELL_90S_LABELS, {"a.jpg": [neighbor("Russell", "90s")], "b.jpg": [neighbor("Champion", "90s")]}
    )
    assert "brand accuracy: 1/2 = 50.0%" in report


def test_main_reports_era_accuracy_separately_from_brand(run_eval):
    report = run_eval(
        RUSSELL_90S_LABELS, {"a.jpg": [neighbor("Russell", "90s")], "b.jpg": [neighbor("Champion", "90s")]}
    )
    assert "era accuracy:   2/2 = 100.0%" in report


def test_main_lists_each_mistake_with_what_was_predicted(run_eval):
    report = run_eval(
        [{"filename": "b.jpg", "brand": "Russell", "era": "90s"}],
        {"b.jpg": [neighbor("Champion", "80s")]},
    )
    assert report.endswith("\nmistakes:\n  b.jpg -> Champion/80s\n")


def test_main_hands_settings_metrics_and_mistakes_to_the_run_log(run_eval, monkeypatch):
    logged = []
    monkeypatch.setattr(
        tracking, "log_eval_run", lambda settings, metrics, mistakes: logged.append((settings, metrics, mistakes))
    )
    run_eval(RUSSELL_90S_LABELS, {"a.jpg": [neighbor("Russell", "90s")], "b.jpg": [neighbor("Champion", "90s")]})
    assert logged == [
        (
            run_settings(library_size=142, test_set_size=2),
            {"brand_accuracy": 0.5, "era_accuracy": 1.0},
            ["b.jpg -> Champion/90s"],
        )
    ]


def test_main_prints_the_same_report_with_tracking_off(run_eval, monkeypatch):
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", "")
    report = run_eval(RUSSELL_90S_LABELS, {"a.jpg": [neighbor("Russell", "90s")], "b.jpg": [neighbor("Russell", "90s")]})
    assert report == "library: 142 photos | test set: 2\nbrand accuracy: 2/2 = 100.0%\nera accuracy:   2/2 = 100.0%\n"
