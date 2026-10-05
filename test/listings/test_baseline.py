import json

import pytest

from listings import baseline, config, tracking
from listings.baseline import baseline_results, target_line_invariants
from listings.parse import prompt_text

EVERY_INVARIANT_PASSES = {
    "starts_with_diamond": True,
    "ends_with_period": True,
    "word_count_ok": True,
    "brand_mentioned": True,
    "era_mentioned": True,
}
BRAND_NOT_MENTIONED = {**EVERY_INVARIANT_PASSES, "brand_mentioned": False}


def exported_example(target: str, brand: str, era: str | None) -> dict:
    facts = {"brand": brand, "era": era, "category": "Hoodies", "gender": "Men", "color": "Grey"}
    return {
        "listing_id": 7,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt_text(facts)}]},
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ],
    }


RUSSELL_EXAMPLE = exported_example("💎 Vintage 90's Boxy Russell Hoodie.", brand="Russell", era="90s")
NIKE_EXAMPLE_WITHOUT_BRAND = exported_example("💎 Vintage Y2K Graphic Tee.", brand="Nike", era="00s")


@pytest.fixture(autouse=True)
def tracking_off(monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", None)


@pytest.fixture
def exported_files(tmp_path, monkeypatch):
    def write_examples(train_examples: list[dict], val_examples: list[dict]):
        for name, examples in (("train.jsonl", train_examples), ("val.jsonl", val_examples)):
            lines = "".join(json.dumps(example, ensure_ascii=False) + "\n" for example in examples)
            (tmp_path / name).write_text(lines, encoding="utf-8")
        monkeypatch.setattr(baseline, "TRAIN_FILE", tmp_path / "train.jsonl")
        monkeypatch.setattr(baseline, "VAL_FILE", tmp_path / "val.jsonl")

    return write_examples


def test_target_line_invariants_scores_the_shops_line_against_the_examples_own_facts():
    assert target_line_invariants(RUSSELL_EXAMPLE) == EVERY_INVARIANT_PASSES


def test_target_line_invariants_flags_a_shop_line_that_leaves_the_brand_out():
    assert target_line_invariants(NIKE_EXAMPLE_WITHOUT_BRAND) == BRAND_NOT_MENTIONED


def test_baseline_results_scores_every_example_in_order():
    assert baseline_results([RUSSELL_EXAMPLE, NIKE_EXAMPLE_WITHOUT_BRAND]) == [
        EVERY_INVARIANT_PASSES,
        BRAND_NOT_MENTIONED,
    ]


def test_main_prints_the_line_count_and_the_summary_over_train_and_val_together(exported_files, capsys):
    exported_files(train_examples=[RUSSELL_EXAMPLE], val_examples=[NIKE_EXAMPLE_WITHOUT_BRAND])
    baseline.main()
    assert capsys.readouterr().out.splitlines() == [
        "shop target lines: 2",
        "starts_with_diamond  2/2 = 100.0%",
        "ends_with_period     2/2 = 100.0%",
        "word_count_ok        2/2 = 100.0%",
        "brand_mentioned      1/2 = 50.0%",
        "era_mentioned        2/2 = 100.0%",
    ]


def test_main_hands_the_results_to_the_baseline_run_log(exported_files, monkeypatch):
    logged = []
    monkeypatch.setattr(tracking, "log_baseline_run", logged.append)
    exported_files(train_examples=[RUSSELL_EXAMPLE], val_examples=[NIKE_EXAMPLE_WITHOUT_BRAND])
    baseline.main()
    assert logged == [[EVERY_INVARIANT_PASSES, BRAND_NOT_MENTIONED]]
