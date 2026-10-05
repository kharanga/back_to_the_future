import json

from listings.parse import (
    INSTRUCTION,
    era_from_line,
    facts_for,
    facts_from_prompt_text,
    first_line,
    normalized_line,
    prompt_text,
)

RUSSELL_HOODIE_ROW = {"brand": "Russell", "category": "Hoodies", "gender": "Men", "color": "Grey"}
RUSSELL_HOODIE_FACTS = {"brand": "Russell", "era": "90s", "category": "Hoodies", "gender": "Men", "color": "Grey"}


def test_first_line_keeps_only_the_diamond_line_of_the_description():
    description = "💎 Vintage 90's Boxy Russell Hoodie.\n📏 Pit to pit 24\n⚠️ No flaws\n⛔️ No returns"
    assert first_line(description) == "💎 Vintage 90's Boxy Russell Hoodie."


def test_first_line_normalizes_the_line_it_keeps():
    assert first_line("💎Vintage Nike Tee\n📏 Pit to pit 22") == "💎 Vintage Nike Tee."


def test_normalized_line_unescapes_html_entities():
    assert normalized_line("💎 Vintage Levi&#39;s 501 Jeans.") == "💎 Vintage Levi's 501 Jeans."


def test_normalized_line_adds_the_missing_space_after_the_diamond():
    assert normalized_line("💎Vintage Nike Tee.") == "💎 Vintage Nike Tee."


def test_normalized_line_collapses_extra_spaces_after_the_diamond():
    assert normalized_line("💎   Vintage Nike Tee.") == "💎 Vintage Nike Tee."


def test_normalized_line_adds_a_period_when_missing():
    assert normalized_line("💎 Vintage Nike Tee") == "💎 Vintage Nike Tee."


def test_normalized_line_keeps_an_existing_period():
    assert normalized_line("💎 Vintage Nike Tee.") == "💎 Vintage Nike Tee."


def test_normalized_line_strips_surrounding_whitespace():
    assert normalized_line("  💎 Vintage Nike Tee.  ") == "💎 Vintage Nike Tee."


def test_normalized_line_leaves_a_curly_apostrophe_as_typed():
    assert normalized_line("💎 Vintage 90’s Nike Tee.") == "💎 Vintage 90’s Nike Tee."


def test_era_from_line_reads_a_decade_with_a_straight_apostrophe():
    assert era_from_line("💎 Vintage 90's Boxy Russell Hoodie.") == "90s"


def test_era_from_line_reads_a_decade_with_a_curly_apostrophe():
    assert era_from_line("💎 Vintage 90’s Boxy Russell Hoodie.") == "90s"


def test_era_from_line_reads_a_decade_with_no_apostrophe():
    assert era_from_line("💎 Vintage 80s Champion Crewneck.") == "80s"


def test_era_from_line_reads_a_four_digit_decade_as_two_digits():
    assert era_from_line("💎 Vintage 1990's Nike Tee.") == "90s"


def test_era_from_line_reads_y2k_as_00s():
    assert era_from_line("💎 Vintage Y2K Baby Tee.") == "00s"


def test_era_from_line_reads_lowercase_y2k_as_00s():
    assert era_from_line("💎 Vintage y2k Baby Tee.") == "00s"


def test_era_from_line_reads_2000s_as_00s():
    assert era_from_line("💎 Vintage 2000's Baby Tee.") == "00s"


def test_era_from_line_is_none_when_the_line_names_no_era():
    assert era_from_line("💎 Vintage Boxy Russell Hoodie.") is None


def test_facts_for_takes_brand_category_gender_and_color_from_the_row_and_era_from_the_line():
    assert facts_for(RUSSELL_HOODIE_ROW, "💎 Vintage 90's Boxy Russell Hoodie.") == RUSSELL_HOODIE_FACTS


def test_facts_for_has_a_null_era_when_the_line_names_none():
    assert facts_for(RUSSELL_HOODIE_ROW, "💎 Vintage Boxy Russell Hoodie.")["era"] is None


def test_facts_for_turns_an_na_color_into_none():
    row = {**RUSSELL_HOODIE_ROW, "color": "N/A"}
    assert facts_for(row, "💎 Vintage 90's Boxy Russell Hoodie.")["color"] is None


def test_facts_for_turns_an_empty_color_into_none():
    row = {**RUSSELL_HOODIE_ROW, "color": ""}
    assert facts_for(row, "💎 Vintage 90's Boxy Russell Hoodie.")["color"] is None


def test_facts_for_keeps_a_missing_color_as_none():
    row = {**RUSSELL_HOODIE_ROW, "color": None}
    assert facts_for(row, "💎 Vintage 90's Boxy Russell Hoodie.")["color"] is None


def test_prompt_text_puts_the_facts_json_on_the_first_line():
    facts_line = prompt_text(RUSSELL_HOODIE_FACTS).split("\n")[0]
    assert json.loads(facts_line) == RUSSELL_HOODIE_FACTS


def test_prompt_text_puts_the_instruction_on_the_second_line():
    assert prompt_text(RUSSELL_HOODIE_FACTS).split("\n")[1] == INSTRUCTION


def test_prompt_text_writes_non_ascii_characters_unescaped():
    facts = {**RUSSELL_HOODIE_FACTS, "brand": "Levi’s"}
    assert '"brand": "Levi’s"' in prompt_text(facts)


def test_facts_from_prompt_text_reads_back_the_facts_prompt_text_wrote():
    assert facts_from_prompt_text(prompt_text(RUSSELL_HOODIE_FACTS)) == RUSSELL_HOODIE_FACTS


def test_facts_from_prompt_text_reads_back_null_facts_as_none():
    facts = {"brand": None, "era": None, "category": "Hoodies", "gender": "Men", "color": None}
    assert facts_from_prompt_text(prompt_text(facts)) == facts
