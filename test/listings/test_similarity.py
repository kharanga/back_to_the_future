import json

import pytest

from listings import config, similarity, tracking
from listings.similarity import (
    exact_match,
    lowest_scoring,
    pair_lines,
    scored_row,
    scored_rows,
    similarity_summary,
    summary_lines,
    word_overlap,
    words,
)

SAME_WORDS_ROW = {"listing_id": 1, "target": "💎 Vintage Nike Tee.", "generated": "💎 Vintage Nike Tee."}
HALF_SHARED_ROW = {"listing_id": 2, "target": "💎 Vintage Nike.", "generated": "💎 Vintage Adidas."}
NOTHING_SHARED_ROW = {"listing_id": 3, "target": "💎 Hoodie.", "generated": "💎 Jeans."}
THREE_ROWS = [SAME_WORDS_ROW, HALF_SHARED_ROW, NOTHING_SHARED_ROW]


@pytest.fixture(autouse=True)
def tracking_off(monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", None)


@pytest.fixture
def eval_file(tmp_path, monkeypatch):
    def write_rows(rows: list[dict]):
        lines = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        (tmp_path / "eval.jsonl").write_text(lines, encoding="utf-8")
        monkeypatch.setattr(similarity, "EVAL_FILE", tmp_path / "eval.jsonl")

    return write_rows


@pytest.fixture
def similarity_run_log(monkeypatch) -> list:
    logged = []
    monkeypatch.setattr(tracking, "log_similarity_run", logged.append)
    return logged


def test_words_are_lowercased_without_the_diamond_or_the_period():
    assert words("💎 Vintage Nike Tee.") == ["vintage", "nike", "tee"]


def test_words_drop_a_straight_apostrophe_from_inside_a_word():
    assert words("90's Levi's") == ["90s", "levis"]


def test_words_treat_a_curly_apostrophe_like_a_straight_one():
    assert words("90’s Levi’s") == words("90's Levi's")


def test_words_read_an_era_with_the_apostrophe_in_front_the_same_way():
    assert words("'00s") == words("00's")


def test_words_split_on_hyphens_and_quotes():
    assert words("“Made in USA” T-Shirt") == ["made", "in", "usa", "t", "shirt"]


def test_exact_match_is_true_when_the_lines_differ_only_in_case_and_punctuation():
    assert exact_match("💎 Vintage 90's Levi's Jeans.", "vintage 90’s levi’s jeans")


def test_exact_match_is_false_when_a_word_differs():
    assert not exact_match("💎 Vintage Nike Tee.", "💎 Vintage Nike Hoodie.")


def test_exact_match_is_false_when_the_same_words_are_in_another_order():
    assert not exact_match("💎 Vintage Nike Tee.", "💎 Nike Vintage Tee.")


def test_word_overlap_is_one_for_the_same_words_in_any_order():
    assert word_overlap("💎 Vintage Nike Tee.", "💎 Nike Vintage Tee.") == 1.0


def test_word_overlap_is_zero_when_no_word_is_shared():
    assert word_overlap("💎 Hoodie.", "💎 Jeans.") == 0.0


def test_word_overlap_is_zero_for_two_empty_lines():
    assert word_overlap("", "") == 0.0


def test_word_overlap_is_the_f1_of_the_shared_words():
    assert word_overlap("💎 Vintage Nike.", "💎 Vintage Adidas.") == 0.5


def test_word_overlap_counts_a_repeated_word_once_per_matching_repeat():
    assert round(word_overlap("a a b", "a b b"), 3) == 0.667


def test_word_overlap_is_high_when_the_generated_line_adds_one_word():
    assert round(word_overlap("💎 Y2K Skull Polo Shirt.", "💎 Y2K Skull Graphic Polo Shirt."), 2) == 0.89


def test_word_overlap_is_low_when_only_the_brand_is_shared():
    shop_line = "💎 Vintage Early 00's Mid Wash Levi's 560 Comfort Fit Jeans."
    assert round(word_overlap(shop_line, "💎 Y2K Levi’s 550 Baggy Jean."), 2) == 0.13


def test_scored_row_is_the_listing_both_lines_the_exact_match_flag_and_the_overlap():
    assert scored_row(HALF_SHARED_ROW) == {
        "listing_id": 2,
        "target": "💎 Vintage Nike.",
        "generated": "💎 Vintage Adidas.",
        "exact_match": False,
        "word_overlap": 0.5,
    }


def test_scored_row_leaves_out_the_other_fields_of_an_eval_row():
    assert "starts_with_diamond" not in scored_row({**HALF_SHARED_ROW, "starts_with_diamond": True})


def test_scored_rows_scores_every_row_in_order():
    assert [row["word_overlap"] for row in scored_rows(THREE_ROWS)] == [1.0, 0.5, 0.0]


def test_similarity_summary_is_the_exact_match_share_the_mean_and_median_overlap_and_the_row_count():
    assert similarity_summary(THREE_ROWS) == {
        "exact_match": 1 / 3,
        "word_overlap_mean": 0.5,
        "word_overlap_median": 0.5,
        "rows": 3,
    }


def test_similarity_summary_raises_when_there_are_no_rows():
    with pytest.raises(ZeroDivisionError):
        similarity_summary([])


def test_lowest_scoring_are_the_rows_with_the_lowest_overlap_lowest_first():
    assert [row["listing_id"] for row in lowest_scoring(THREE_ROWS, 2)] == [3, 2]


def test_lowest_scoring_gives_five_rows_unless_told_otherwise():
    seven_rows = [{**HALF_SHARED_ROW, "listing_id": listing_id} for listing_id in range(7)]
    assert len(lowest_scoring(seven_rows)) == 5


def test_summary_lines_are_the_exact_match_count_and_the_mean_and_median_overlap():
    summary = {"exact_match": 17 / 160, "word_overlap_mean": 0.6321, "word_overlap_median": 0.6154, "rows": 160}
    assert summary_lines(summary) == [
        "exact match          17/160 = 10.6%",
        "word overlap mean    0.632",
        "word overlap median  0.615",
    ]


def test_pair_lines_show_the_shop_line_the_generated_line_and_the_overlap():
    assert pair_lines(scored_row(HALF_SHARED_ROW)) == [
        "    2  shop:      💎 Vintage Nike.",
        "       generated: 💎 Vintage Adidas.",
        "       overlap:   0.50",
        "",
    ]


def test_main_prints_the_summary_then_the_lowest_pairs_lowest_first(eval_file, similarity_run_log, capsys):
    eval_file(THREE_ROWS)
    similarity.main()
    assert capsys.readouterr().out.splitlines() == [
        "exact match          1/3 = 33.3%",
        "word overlap mean    0.500",
        "word overlap median  0.500",
        "",
        "lowest 5 by word overlap:",
        "",
        *pair_lines(scored_row(NOTHING_SHARED_ROW)),
        *pair_lines(scored_row(HALF_SHARED_ROW)),
        *pair_lines(scored_row(SAME_WORDS_ROW)),
    ]


def test_main_hands_the_summary_to_the_similarity_run_log(eval_file, similarity_run_log):
    eval_file(THREE_ROWS)
    similarity.main()
    assert similarity_run_log == [similarity_summary(THREE_ROWS)]


def test_main_only_prints_when_tracking_is_off(eval_file, capsys):
    eval_file(THREE_ROWS)
    similarity.main()
    assert capsys.readouterr().out.startswith("exact match          1/3 = 33.3%")


def test_main_raises_before_logging_when_the_eval_file_has_no_rows(eval_file, similarity_run_log):
    eval_file([])
    with pytest.raises(ZeroDivisionError):
        similarity.main()
    assert similarity_run_log == []
