import pytest

from listings.invariants import (
    BLANK_TAG_BRANDS,
    brand_mentioned,
    era_mentioned,
    invariant_pass_rates,
    invariants,
    pass_count,
    print_summary,
    straight_apostrophes,
)

RUSSELL_90S = {"brand": "Russell", "era": "90s"}
EVERY_INVARIANT_PASSES = {
    "starts_with_diamond": True,
    "ends_with_period": True,
    "word_count_ok": True,
    "brand_mentioned": True,
    "era_mentioned": True,
}
BRAND_NOT_MENTIONED = {**EVERY_INVARIANT_PASSES, "brand_mentioned": False}
TWELVE_WORD_LINE = "💎 Vintage 90's Russell Hoodie two three four five six seven eight."
THIRTEEN_WORD_LINE = "💎 Vintage 90's Russell Hoodie two three four five six seven eight nine."


def test_straight_apostrophes_replaces_a_curly_apostrophe_with_a_straight_one():
    assert straight_apostrophes("levi’s") == "levi's"


def test_straight_apostrophes_lowercases_the_text():
    assert straight_apostrophes("Levi's") == "levi's"


def test_brand_mentioned_is_true_when_the_line_names_the_brand():
    assert brand_mentioned("💎 Vintage 90's Boxy Russell Hoodie.", "Russell")


def test_brand_mentioned_is_false_when_the_line_leaves_the_brand_out():
    assert not brand_mentioned("💎 Vintage 90's Boxy Hoodie.", "Russell")


def test_brand_mentioned_ignores_case():
    assert brand_mentioned("💎 Vintage 90's Boxy RUSSELL Hoodie.", "Russell")


def test_brand_mentioned_is_true_when_one_word_of_the_brand_is_in_the_line():
    assert brand_mentioned("💎 Vintage 90's Tommy Jeans.", "Tommy Hilfiger")


def test_brand_mentioned_ignores_brand_words_shorter_than_three_letters():
    assert not brand_mentioned("💎 Vintage 90's LA Tee.", "LA Gear")


def test_brand_mentioned_matches_a_straight_apostrophe_brand_to_a_curly_apostrophe_line():
    assert brand_mentioned("💎 Vintage Levi’s 550 Jeans.", "Levi's")


def test_brand_mentioned_matches_a_curly_apostrophe_brand_to_a_straight_apostrophe_line():
    assert brand_mentioned("💎 Vintage Levi's 550 Jeans.", "Levi’s")


def test_brand_mentioned_passes_when_the_brand_is_none():
    assert brand_mentioned("💎 Vintage 90's Boxy Hoodie.", None)


def test_brand_mentioned_passes_when_the_brand_is_empty():
    assert brand_mentioned("💎 Vintage 90's Boxy Hoodie.", "")


def test_brand_mentioned_passes_when_the_brand_is_other():
    assert brand_mentioned("💎 Vintage 90's Boxy Hoodie.", "Other")


@pytest.mark.parametrize("blank_tag_brand", sorted(BLANK_TAG_BRANDS))
def test_brand_mentioned_passes_for_a_blank_tag_brand_the_shop_leaves_out_of_titles(blank_tag_brand):
    assert brand_mentioned("💎 Vintage 90's Harley Davidson Graphic Tee.", blank_tag_brand)


def test_brand_mentioned_counts_hanes_gildan_and_jansport_as_blank_tag_brands():
    assert {"Hanes", "Gildan", "Jansport"} <= BLANK_TAG_BRANDS


def test_era_mentioned_is_true_when_the_line_names_the_era():
    assert era_mentioned("💎 Vintage 90's Boxy Russell Hoodie.", "90s")


def test_era_mentioned_is_false_when_the_line_names_another_era():
    assert not era_mentioned("💎 Vintage 80's Boxy Russell Hoodie.", "90s")


def test_era_mentioned_is_false_when_the_line_names_no_era():
    assert not era_mentioned("💎 Vintage Boxy Russell Hoodie.", "90s")


def test_era_mentioned_reads_y2k_in_the_line_as_00s():
    assert era_mentioned("💎 Vintage Y2K Baby Tee.", "00s")


def test_era_mentioned_passes_when_the_era_fact_is_none():
    assert era_mentioned("💎 Vintage Boxy Russell Hoodie.", None)


def test_invariants_returns_the_five_flags_all_true_for_a_line_in_the_shops_format():
    assert invariants("💎 Vintage 90's Boxy Russell Hoodie.", RUSSELL_90S) == EVERY_INVARIANT_PASSES


def test_invariants_fails_starts_with_diamond_without_the_diamond():
    assert not invariants("Vintage 90's Boxy Russell Hoodie.", RUSSELL_90S)["starts_with_diamond"]


def test_invariants_fails_ends_with_period_without_the_period():
    assert not invariants("💎 Vintage 90's Boxy Russell Hoodie", RUSSELL_90S)["ends_with_period"]


def test_invariants_passes_word_count_ok_at_twelve_words():
    assert invariants(TWELVE_WORD_LINE, RUSSELL_90S)["word_count_ok"]


def test_invariants_fails_word_count_ok_above_twelve_words():
    assert not invariants(THIRTEEN_WORD_LINE, RUSSELL_90S)["word_count_ok"]


def test_invariants_fails_brand_mentioned_when_the_line_leaves_the_brand_out():
    assert not invariants("💎 Vintage 90's Boxy Hoodie.", RUSSELL_90S)["brand_mentioned"]


def test_invariants_fails_era_mentioned_when_the_line_leaves_the_era_out():
    assert not invariants("💎 Vintage Boxy Russell Hoodie.", RUSSELL_90S)["era_mentioned"]


def test_pass_count_counts_the_rows_that_pass_one_invariant():
    assert pass_count([EVERY_INVARIANT_PASSES, BRAND_NOT_MENTIONED], "brand_mentioned") == 1


def test_invariant_pass_rates_is_the_passing_fraction_for_each_invariant():
    assert invariant_pass_rates([EVERY_INVARIANT_PASSES, BRAND_NOT_MENTIONED]) == {
        "starts_with_diamond": 1.0,
        "ends_with_period": 1.0,
        "word_count_ok": 1.0,
        "brand_mentioned": 0.5,
        "era_mentioned": 1.0,
    }


def test_invariant_pass_rates_raises_when_there_are_no_rows_to_score():
    with pytest.raises(ZeroDivisionError):
        invariant_pass_rates([])


def test_print_summary_prints_count_and_percentage_for_each_invariant(capsys):
    print_summary([EVERY_INVARIANT_PASSES, BRAND_NOT_MENTIONED])
    assert capsys.readouterr().out.splitlines() == [
        "starts_with_diamond  2/2 = 100.0%",
        "ends_with_period     2/2 = 100.0%",
        "word_count_ok        2/2 = 100.0%",
        "brand_mentioned      1/2 = 50.0%",
        "era_mentioned        2/2 = 100.0%",
    ]


def test_print_summary_raises_when_there_are_no_rows_to_score():
    with pytest.raises(ZeroDivisionError):
        print_summary([])
