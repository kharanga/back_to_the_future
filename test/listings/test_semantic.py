import sys
from types import SimpleNamespace

import pytest

from listings.semantic import (
    cosine_similarity,
    embedded,
    load_embedder,
    lowest_scoring,
    mismatched_scores,
    pair_lines,
    rotated_by_one,
    semantic_rows,
    semantic_score,
    semantic_summary,
    summary_lines,
    text_to_embed,
    vector_length,
    with_mismatched_score,
)

HOODIE_ROW = {"listing_id": 1, "target": "💎 Hoodie.", "generated": "💎 Grey Hoodie."}
JEANS_ROW = {"listing_id": 2, "target": "💎 Jeans.", "generated": "💎 Tee."}
JACKET_ROW = {"listing_id": 3, "target": "💎 Jacket.", "generated": "💎 Coat."}
THREE_ROWS = [HOODIE_ROW, JEANS_ROW, JACKET_ROW]
VECTOR_FOR_TEXT = {
    "Hoodie.": [1, 0],
    "Grey Hoodie.": [1, 0],
    "Jeans.": [0, 1],
    "Tee.": [1, 0],
    "Jacket.": [3, 4],
    "Coat.": [4, 3],
}


class FakeEmbedder:
    def __init__(self):
        self.encode_calls = []

    def encode(self, texts, normalize_embeddings):
        self.encode_calls.append((texts, normalize_embeddings))
        return [VECTOR_FOR_TEXT[text] for text in texts]


def test_load_embedder_loads_the_minilm_sentence_transformer(monkeypatch):
    fake_library = SimpleNamespace(SentenceTransformer=lambda model_name: f"embedder for {model_name}")
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_library)
    assert load_embedder() == "embedder for sentence-transformers/all-MiniLM-L6-v2"


def test_load_embedder_is_the_only_thing_that_imports_sentence_transformers():
    assert "sentence_transformers" not in sys.modules


def test_text_to_embed_drops_the_diamond_and_surrounding_spaces():
    assert text_to_embed(" 💎 Vintage Nike Tee. ") == "Vintage Nike Tee."


def test_text_to_embed_turns_curly_apostrophes_into_straight_ones():
    assert text_to_embed("💎 Vintage 90’s Levi’s Jeans.") == "Vintage 90's Levi's Jeans."


def test_text_to_embed_keeps_the_period_and_other_punctuation():
    assert text_to_embed("💎 Vintage “501” T-Shirt.") == "Vintage “501” T-Shirt."


def test_embedded_is_one_vector_per_line():
    assert embedded(FakeEmbedder(), ["💎 Hoodie.", "💎 Jeans."]) == [[1, 0], [0, 1]]


def test_embedded_encodes_every_line_in_one_call_asking_for_normalized_vectors():
    embedder = FakeEmbedder()
    embedded(embedder, ["💎 Hoodie.", "💎 Jeans."])
    assert embedder.encode_calls == [(["Hoodie.", "Jeans."], True)]


def test_vector_length_is_the_euclidean_length():
    assert vector_length([3, 4]) == 5.0


def test_cosine_similarity_is_one_for_vectors_pointing_the_same_way():
    assert cosine_similarity([3, 4], [6, 8]) == pytest.approx(1.0)


def test_cosine_similarity_is_zero_for_vectors_at_a_right_angle():
    assert cosine_similarity([1, 0], [0, 1]) == 0.0


def test_cosine_similarity_is_negative_for_vectors_pointing_opposite_ways():
    assert cosine_similarity([1, 0], [-1, 0]) == -1.0


def test_cosine_similarity_is_zero_when_a_vector_has_no_length():
    assert cosine_similarity([0, 0], [3, 4]) == 0.0


def test_semantic_score_is_one_hundred_for_identical_vectors():
    assert semantic_score([3, 4], [3, 4]) == pytest.approx(100.0)


def test_semantic_score_is_the_cosine_similarity_times_one_hundred():
    assert semantic_score([3, 4], [4, 3]) == pytest.approx(96.0)


def test_semantic_score_is_zero_for_unrelated_vectors():
    assert semantic_score([1, 0], [0, 1]) == 0.0


def test_semantic_score_is_never_below_zero():
    assert semantic_score([1, 0], [-1, 0]) == 0.0


def test_rotated_by_one_moves_the_first_item_to_the_end():
    assert rotated_by_one([1, 2, 3]) == [2, 3, 1]


def test_mismatched_scores_pair_each_generated_line_with_the_next_rows_shop_line():
    target_vectors = [[1, 0], [0, 1], [3, 4]]
    generated_vectors = [[1, 0], [1, 0], [4, 3]]
    assert mismatched_scores(target_vectors, generated_vectors) == pytest.approx([0.0, 60.0, 80.0])


def test_mismatched_scores_are_empty_for_a_single_row():
    assert mismatched_scores([[1, 0]], [[1, 0]]) == []


def test_with_mismatched_score_adds_the_rows_own_mismatched_score():
    assert with_mismatched_score({"listing_id": 2}, [0.0, 60.0, 80.0], 1) == {
        "listing_id": 2,
        "semantic_mismatched": 60.0,
    }


def test_with_mismatched_score_leaves_the_row_alone_when_there_are_no_mismatched_scores():
    assert with_mismatched_score({"listing_id": 1}, [], 0) == {"listing_id": 1}


def test_semantic_rows_add_each_rows_score_against_its_own_shop_line():
    scores = [row["semantic"] for row in semantic_rows(FakeEmbedder(), THREE_ROWS)]
    assert scores == pytest.approx([100.0, 0.0, 96.0])


def test_semantic_rows_add_each_rows_score_against_another_rows_shop_line():
    scores = [row["semantic_mismatched"] for row in semantic_rows(FakeEmbedder(), THREE_ROWS)]
    assert scores == pytest.approx([0.0, 60.0, 80.0])


def test_semantic_rows_keep_the_fields_of_the_row():
    assert semantic_rows(FakeEmbedder(), THREE_ROWS)[0]["target"] == "💎 Hoodie."


def test_semantic_rows_embed_the_shop_lines_in_one_call_and_the_generated_lines_in_another():
    embedder = FakeEmbedder()
    semantic_rows(embedder, THREE_ROWS)
    assert embedder.encode_calls == [
        (["Hoodie.", "Jeans.", "Jacket."], True),
        (["Grey Hoodie.", "Tee.", "Coat."], True),
    ]


def test_semantic_rows_leave_out_the_mismatched_score_for_a_single_row():
    assert "semantic_mismatched" not in semantic_rows(FakeEmbedder(), [HOODIE_ROW])[0]


def test_semantic_summary_is_the_mean_and_median_score_and_the_mismatched_mean():
    scored_rows = [
        {"semantic": 100.0, "semantic_mismatched": 0.0},
        {"semantic": 0.0, "semantic_mismatched": 60.0},
        {"semantic": 80.0, "semantic_mismatched": 90.0},
    ]
    assert semantic_summary(scored_rows) == {
        "semantic_mean": 60.0,
        "semantic_median": 80.0,
        "semantic_mismatched_mean": 50.0,
    }


def test_semantic_summary_leaves_out_the_mismatched_mean_for_a_single_row():
    assert semantic_summary([{"semantic": 100.0}]) == {"semantic_mean": 100.0, "semantic_median": 100.0}


def test_lowest_scoring_are_the_rows_with_the_lowest_semantic_score_lowest_first():
    scored_rows = [{"listing_id": 1, "semantic": 100.0}, {"listing_id": 2, "semantic": 0.0}, {"listing_id": 3, "semantic": 96.0}]
    assert [row["listing_id"] for row in lowest_scoring(scored_rows, 2)] == [2, 3]


def test_lowest_scoring_gives_five_rows_unless_told_otherwise():
    seven_rows = [{"listing_id": listing_id, "semantic": 50.0} for listing_id in range(7)]
    assert len(lowest_scoring(seven_rows)) == 5


def test_summary_lines_are_the_mean_out_of_one_hundred_the_median_and_the_mismatched_mean():
    summary = {"semantic_mean": 84.21, "semantic_median": 84.45, "semantic_mismatched_mean": 69.08}
    assert summary_lines(summary) == [
        "semantic mean        84.2 out of 100",
        "semantic median      84.5",
        "semantic mismatched  69.1 (each line against another row's shop line)",
    ]


def test_summary_lines_leave_out_the_mismatched_line_when_there_is_no_mismatched_mean():
    assert summary_lines({"semantic_mean": 100.0, "semantic_median": 100.0}) == [
        "semantic mean        100.0 out of 100",
        "semantic median      100.0",
    ]


def test_pair_lines_show_the_shop_line_the_generated_line_and_the_semantic_score():
    assert pair_lines({**JACKET_ROW, "semantic": 96.0}) == [
        "    3  shop:      💎 Jacket.",
        "       generated: 💎 Coat.",
        "       semantic:  96.0",
        "",
    ]
