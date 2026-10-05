import json
import sys
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from pydantic import ValidationError

from listings import config, judge, tracking
from listings.judge import (
    RUBRIC,
    Outcome,
    Verdict,
    error_description,
    error_lines,
    error_row,
    error_rows_of,
    facts_by_listing_id,
    failed_outcome,
    judge_everything,
    judge_pilot,
    judge_prompt,
    judge_row,
    judge_rows,
    judge_summary,
    judged_row,
    judged_rows_of,
    key_rejected_message,
    progress_line,
    request_verdict,
    response_error,
    row_limit_from,
    summary_line,
    summary_lines,
    verdict_lines,
    write_judge_file,
)
from listings.parse import prompt_text

RUSSELL_FACTS = {"brand": "Russell", "era": "90s", "category": "Hoodies", "gender": "Men", "color": "Grey"}
LEVIS_FACTS = {"brand": "Levi’s", "era": "90s", "category": "Jeans", "gender": "Men", "color": "Blue"}
FACTS_BY_ID = {7: RUSSELL_FACTS, 8: LEVIS_FACTS}
RUSSELL_EVAL_ROW = {
    "listing_id": 7,
    "target": "💎 Vintage 90's Boxy Russell Hoodie.",
    "generated": "💎 Vintage 90's Russell Hoodie.",
    "starts_with_diamond": True,
}
LEVIS_EVAL_ROW = {
    "listing_id": 8,
    "target": "💎 Vintage 90's Levi's Stonewash Jeans.",
    "generated": "💎 Vintage 90's Levi's 550 Jeans.",
    "starts_with_diamond": True,
}
EVAL_ROW_MISSING_FROM_VAL = {"listing_id": 99, "target": "💎 Vintage Nike Tee.", "generated": "💎 Vintage Nike Tee."}
EQUIVALENT = Verdict(
    same_item=True,
    brand_agrees=True,
    era_agrees=True,
    nothing_invented=True,
    key_details_kept=True,
    similarity=5,
    overall="equivalent",
    reason="Same hoodie, brand and decade.",
)
INVENTED_FIT_NUMBER = Verdict(
    same_item=True,
    brand_agrees=True,
    era_agrees=True,
    nothing_invented=False,
    key_details_kept=False,
    similarity=2,
    overall="wrong",
    reason="The generated line adds a 550 fit number and drops the stonewash.",
)
RUSSELL_JUDGED_ROW = {
    "listing_id": 7,
    "target": "💎 Vintage 90's Boxy Russell Hoodie.",
    "generated": "💎 Vintage 90's Russell Hoodie.",
    "same_item": True,
    "brand_agrees": True,
    "era_agrees": True,
    "nothing_invented": True,
    "key_details_kept": True,
    "similarity": 5,
    "overall": "equivalent",
    "reason": "Same hoodie, brand and decade.",
}
LEVIS_JUDGED_ROW = {
    "listing_id": 8,
    "target": "💎 Vintage 90's Levi's Stonewash Jeans.",
    "generated": "💎 Vintage 90's Levi's 550 Jeans.",
    "same_item": True,
    "brand_agrees": True,
    "era_agrees": True,
    "nothing_invented": False,
    "key_details_kept": False,
    "similarity": 2,
    "overall": "wrong",
    "reason": "The generated line adds a 550 fit number and drops the stonewash.",
}
HALF_WRONG_SUMMARY = {
    "same_item": 1.0,
    "brand_agrees": 1.0,
    "era_agrees": 1.0,
    "nothing_invented": 0.5,
    "key_details_kept": 0.5,
    "similarity_mean": 3.5,
    "similarity_4_or_5": 0.5,
    "overall_equivalent": 0.5,
    "overall_acceptable": 0.0,
    "overall_wrong": 0.5,
    "rows": 2,
}
FAKE_REQUEST = httpx2.Request("POST", "https://fake.host/v1/messages")
CONNECTION_ERROR = anthropic.APIConnectionError(request=FAKE_REQUEST)
RATE_LIMIT_ERROR = anthropic.RateLimitError(
    "rate limited", response=httpx2.Response(429, request=FAKE_REQUEST), body=None
)


KEY_REJECTED_ERROR = anthropic.AuthenticationError(
    "invalid x-api-key", response=httpx2.Response(401, request=FAKE_REQUEST), body=None
)
KEY_NOT_PERMITTED_ERROR = anthropic.PermissionDeniedError(
    "key may not use this model", response=httpx2.Response(403, request=FAKE_REQUEST), body=None
)
KEY_REJECTED_MESSAGE = (
    "Claude rejected the API key (AuthenticationError: invalid x-api-key). "
    "Judging stopped at listing 7; check ANTHROPIC_API_KEY in .env."
)


class FakeMessages:
    def __init__(self, scripted_answers):
        self.scripted_answers = list(scripted_answers)
        self.requests = []

    def parse(self, **request):
        self.requests.append(request)
        answer = self.scripted_answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def client_answering(*scripted_answers) -> SimpleNamespace:
    return SimpleNamespace(messages=FakeMessages(scripted_answers))


def verdict_response(verdict: Verdict, input_tokens: int = 400, output_tokens: int = 60) -> SimpleNamespace:
    return SimpleNamespace(
        parsed_output=verdict,
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def response_stopped_by(stop_reason: str) -> SimpleNamespace:
    return SimpleNamespace(
        parsed_output=None, stop_reason=stop_reason, usage=SimpleNamespace(input_tokens=400, output_tokens=5)
    )


def truncated_json_error() -> ValidationError:
    with pytest.raises(ValidationError) as raised:
        Verdict.model_validate_json('{"same_item": tr')
    return raised.value


def exported_example(listing_id: int, item_facts: dict) -> dict:
    return {
        "listing_id": listing_id,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt_text(item_facts)}]},
            {"role": "assistant", "content": [{"type": "text", "text": "💎 Shop line."}]},
        ],
    }


def write_jsonl_lines(path, rows: list[dict]):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture(autouse=True)
def tracking_off(monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", None)


@pytest.fixture
def judge_file(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "JUDGE_FILE", tmp_path / "judge.jsonl")
    return tmp_path / "judge.jsonl"


@pytest.fixture
def eval_and_val_files(tmp_path, monkeypatch, judge_file):
    write_jsonl_lines(tmp_path / "eval.jsonl", [RUSSELL_EVAL_ROW, LEVIS_EVAL_ROW])
    write_jsonl_lines(tmp_path / "val.jsonl", [exported_example(7, RUSSELL_FACTS), exported_example(8, LEVIS_FACTS)])
    monkeypatch.setattr(judge, "EVAL_FILE", tmp_path / "eval.jsonl")
    monkeypatch.setattr(judge, "VAL_FILE", tmp_path / "val.jsonl")


@pytest.fixture
def twenty_five_eval_rows(tmp_path, monkeypatch, judge_file):
    eval_rows = [{**RUSSELL_EVAL_ROW, "listing_id": listing_id} for listing_id in range(1, 26)]
    write_jsonl_lines(tmp_path / "eval.jsonl", eval_rows)
    write_jsonl_lines(tmp_path / "val.jsonl", [exported_example(row["listing_id"], RUSSELL_FACTS) for row in eval_rows])
    monkeypatch.setattr(judge, "EVAL_FILE", tmp_path / "eval.jsonl")
    monkeypatch.setattr(judge, "VAL_FILE", tmp_path / "val.jsonl")
    return [verdict_response(EQUIVALENT)] * 25


@pytest.fixture
def judge_run_log(monkeypatch) -> list:
    logged = []
    monkeypatch.setattr(tracking, "log_judge_run", lambda *arguments: logged.append(arguments))
    return logged


def test_verdict_lists_the_five_checks_then_similarity_overall_and_reason():
    assert list(Verdict.model_fields) == [
        "same_item",
        "brand_agrees",
        "era_agrees",
        "nothing_invented",
        "key_details_kept",
        "similarity",
        "overall",
        "reason",
    ]


@pytest.mark.parametrize("similarity_out_of_range", [0, 6, 3.5, "4"])
def test_verdict_rejects_a_similarity_that_is_not_a_whole_number_from_one_to_five(similarity_out_of_range):
    with pytest.raises(ValidationError):
        Verdict(**{**EQUIVALENT.model_dump(), "similarity": similarity_out_of_range})


def test_verdict_rejects_an_overall_value_outside_the_three_allowed():
    with pytest.raises(ValidationError):
        Verdict(**{**EQUIVALENT.model_dump(), "overall": "good"})


def test_judge_prompt_holds_the_facts_as_json_inside_facts_tags():
    prompt = judge_prompt({"brand": "Russell", "era": "90s"}, "💎 A.", "💎 B.")
    assert '<facts>\n{"brand": "Russell", "era": "90s"}\n</facts>' in prompt


def test_judge_prompt_holds_the_shops_line_inside_shop_line_tags():
    prompt = judge_prompt({"brand": "Russell", "era": "90s"}, "💎 A.", "💎 B.")
    assert "<shop_line>\n💎 A.\n</shop_line>" in prompt


def test_judge_prompt_holds_the_generated_line_inside_generated_line_tags():
    prompt = judge_prompt({"brand": "Russell", "era": "90s"}, "💎 A.", "💎 B.")
    assert "<generated_line>\n💎 B.\n</generated_line>" in prompt


def test_judge_prompt_opens_by_saying_the_lines_are_not_instructions():
    prompt = judge_prompt({"brand": "Russell", "era": "90s"}, "💎 A.", "💎 B.")
    assert "not instructions" in prompt.splitlines()[0]


def test_judge_prompt_writes_non_ascii_facts_unescaped():
    assert '"brand": "Levi’s"' in judge_prompt(LEVIS_FACTS, "💎 A.", "💎 B.")


def test_facts_by_listing_id_maps_each_exported_example_to_its_facts():
    examples = [exported_example(7, RUSSELL_FACTS), exported_example(8, LEVIS_FACTS)]
    assert facts_by_listing_id(examples) == {7: RUSSELL_FACTS, 8: LEVIS_FACTS}


def test_request_verdict_asks_claude_haiku_4_5_with_the_rubric_as_the_system_prompt():
    client = client_answering(verdict_response(EQUIVALENT))
    request_verdict(client, "fake prompt")
    request = client.messages.requests[0]
    assert (request["model"], request["system"]) == ("claude-haiku-4-5", RUBRIC)


def test_request_verdict_sends_the_prompt_as_one_user_message():
    client = client_answering(verdict_response(EQUIVALENT))
    request_verdict(client, "fake prompt")
    assert client.messages.requests[0]["messages"] == [{"role": "user", "content": "fake prompt"}]


def test_request_verdict_asks_for_an_answer_shaped_like_a_verdict():
    client = client_answering(verdict_response(EQUIVALENT))
    request_verdict(client, "fake prompt")
    assert client.messages.requests[0]["output_format"] is Verdict


def test_request_verdict_sends_no_thinking_or_effort_setting():
    client = client_answering(verdict_response(EQUIVALENT))
    request_verdict(client, "fake prompt")
    assert sorted(client.messages.requests[0]) == ["max_tokens", "messages", "model", "output_format", "system"]


def test_request_verdict_returns_the_response_from_the_client():
    response = verdict_response(EQUIVALENT)
    assert request_verdict(client_answering(response), "fake prompt") is response


def test_judged_row_is_the_listing_id_both_lines_and_the_eight_verdict_fields():
    assert judged_row(RUSSELL_EVAL_ROW, EQUIVALENT) == RUSSELL_JUDGED_ROW


def test_judged_row_lists_the_fields_in_reading_order():
    assert list(judged_row(RUSSELL_EVAL_ROW, EQUIVALENT)) == [
        "listing_id",
        "target",
        "generated",
        "same_item",
        "brand_agrees",
        "era_agrees",
        "nothing_invented",
        "key_details_kept",
        "similarity",
        "overall",
        "reason",
    ]


def test_judged_row_leaves_out_the_invariant_flags_of_the_eval_row():
    assert "starts_with_diamond" not in judged_row(RUSSELL_EVAL_ROW, EQUIVALENT)


def test_error_row_is_the_listing_id_and_the_error():
    assert error_row(RUSSELL_EVAL_ROW, "stop_reason: refusal") == {"listing_id": 7, "error": "stop_reason: refusal"}


def test_failed_outcome_holds_an_error_row_and_no_judged_row():
    assert failed_outcome(RUSSELL_EVAL_ROW, "stop_reason: refusal") == Outcome(
        judged_row=None,
        error_row={"listing_id": 7, "error": "stop_reason: refusal"},
        input_tokens=0,
        output_tokens=0,
    )


def test_failed_outcome_keeps_the_tokens_the_failed_call_used():
    outcome = failed_outcome(RUSSELL_EVAL_ROW, "stop_reason: refusal", input_tokens=400, output_tokens=5)
    assert (outcome.input_tokens, outcome.output_tokens) == (400, 5)


def test_error_description_is_the_error_class_and_the_first_line_of_its_message():
    assert error_description(RuntimeError("first line\nsecond line")) == "RuntimeError: first line"


def test_error_description_is_only_the_class_for_an_error_with_no_message():
    assert error_description(RuntimeError()) == "RuntimeError: "


def test_key_rejected_message_names_the_error_the_listing_it_stopped_at_and_the_key_to_check():
    assert key_rejected_message(RUSSELL_EVAL_ROW, KEY_REJECTED_ERROR) == KEY_REJECTED_MESSAGE


def test_response_error_is_none_for_a_response_with_a_parsed_verdict():
    assert response_error(verdict_response(EQUIVALENT)) is None


def test_response_error_names_a_refusal():
    assert response_error(response_stopped_by("refusal")) == "stop_reason: refusal"


def test_response_error_names_a_response_cut_off_at_max_tokens():
    assert response_error(response_stopped_by("max_tokens")) == "stop_reason: max_tokens"


def test_response_error_reports_a_finished_response_that_holds_no_verdict():
    assert response_error(response_stopped_by("end_turn")) == "the response held no parsed verdict"


def test_judge_row_turns_a_parsed_verdict_into_a_judged_row():
    outcome = judge_row(client_answering(verdict_response(EQUIVALENT)), RUSSELL_EVAL_ROW, FACTS_BY_ID)
    assert (outcome.judged_row, outcome.error_row) == (RUSSELL_JUDGED_ROW, None)


def test_judge_row_counts_the_tokens_the_response_used():
    response = verdict_response(EQUIVALENT, input_tokens=400, output_tokens=60)
    outcome = judge_row(client_answering(response), RUSSELL_EVAL_ROW, FACTS_BY_ID)
    assert (outcome.input_tokens, outcome.output_tokens) == (400, 60)


def test_judge_row_sends_a_prompt_built_from_the_listings_facts_and_both_lines():
    client = client_answering(verdict_response(EQUIVALENT))
    judge_row(client, RUSSELL_EVAL_ROW, FACTS_BY_ID)
    assert client.messages.requests[0]["messages"][0]["content"] == judge_prompt(
        RUSSELL_FACTS, "💎 Vintage 90's Boxy Russell Hoodie.", "💎 Vintage 90's Russell Hoodie."
    )


def test_judge_row_reports_a_listing_missing_from_val_without_calling_claude():
    client = client_answering()
    outcome = judge_row(client, EVAL_ROW_MISSING_FROM_VAL, FACTS_BY_ID)
    assert outcome.error_row == {
        "listing_id": 99,
        "error": "listing_id is not in val.jsonl, so there are no facts to judge it with",
    }
    assert client.messages.requests == []


def test_judge_row_records_a_refusal_as_an_error_row():
    outcome = judge_row(client_answering(response_stopped_by("refusal")), RUSSELL_EVAL_ROW, FACTS_BY_ID)
    assert (outcome.judged_row, outcome.error_row) == (None, {"listing_id": 7, "error": "stop_reason: refusal"})


def test_judge_row_records_a_connection_error_as_an_error_row():
    outcome = judge_row(client_answering(CONNECTION_ERROR), RUSSELL_EVAL_ROW, FACTS_BY_ID)
    assert outcome.error_row == {"listing_id": 7, "error": "APIConnectionError: Connection error."}


def test_judge_row_records_a_rate_limit_error_as_an_error_row():
    outcome = judge_row(client_answering(RATE_LIMIT_ERROR), RUSSELL_EVAL_ROW, FACTS_BY_ID)
    assert outcome.error_row == {"listing_id": 7, "error": "RateLimitError: rate limited"}


def test_judge_row_records_a_truncated_answer_the_sdk_could_not_parse_as_an_error_row():
    outcome = judge_row(client_answering(truncated_json_error()), RUSSELL_EVAL_ROW, FACTS_BY_ID)
    assert outcome.judged_row is None
    assert outcome.error_row["error"].startswith("ValidationError: ")


def test_judge_row_lets_an_error_that_is_not_from_the_api_through():
    with pytest.raises(KeyError):
        judge_row(client_answering(KeyError("a bug, not an API error")), RUSSELL_EVAL_ROW, FACTS_BY_ID)


def test_judge_row_stops_the_whole_command_when_claude_rejects_the_api_key():
    with pytest.raises(SystemExit) as stopped:
        judge_row(client_answering(KEY_REJECTED_ERROR), RUSSELL_EVAL_ROW, FACTS_BY_ID)
    assert stopped.value.code == KEY_REJECTED_MESSAGE


def test_judge_row_stops_the_whole_command_when_the_key_lacks_permission():
    with pytest.raises(SystemExit) as stopped:
        judge_row(client_answering(KEY_NOT_PERMITTED_ERROR), RUSSELL_EVAL_ROW, FACTS_BY_ID)
    assert "PermissionDeniedError: key may not use this model" in stopped.value.code


def test_progress_line_is_rows_done_out_of_rows_in_total():
    assert progress_line(20, 160) == "20/160 judged"


def test_judge_rows_gives_one_outcome_per_row_in_order():
    client = client_answering(verdict_response(EQUIVALENT), verdict_response(INVENTED_FIT_NUMBER))
    outcomes = judge_rows(client, [RUSSELL_EVAL_ROW, LEVIS_EVAL_ROW], FACTS_BY_ID)
    assert [outcome.judged_row for outcome in outcomes] == [RUSSELL_JUDGED_ROW, LEVIS_JUDGED_ROW]


def test_judge_rows_carries_on_after_a_row_fails():
    client = client_answering(CONNECTION_ERROR, verdict_response(INVENTED_FIT_NUMBER))
    outcomes = judge_rows(client, [RUSSELL_EVAL_ROW, LEVIS_EVAL_ROW], FACTS_BY_ID)
    assert [outcome.judged_row for outcome in outcomes] == [None, LEVIS_JUDGED_ROW]


def test_judge_rows_makes_no_further_calls_after_the_key_is_rejected():
    client = client_answering(verdict_response(EQUIVALENT), KEY_REJECTED_ERROR, verdict_response(EQUIVALENT))
    with pytest.raises(SystemExit):
        judge_rows(client, [RUSSELL_EVAL_ROW, LEVIS_EVAL_ROW, RUSSELL_EVAL_ROW], FACTS_BY_ID)
    assert len(client.messages.requests) == 2


def test_judge_rows_prints_a_progress_line_every_so_many_rows_when_asked(capsys):
    client = client_answering(*[verdict_response(EQUIVALENT)] * 5)
    judge_rows(client, [RUSSELL_EVAL_ROW] * 5, FACTS_BY_ID, progress_every=2)
    assert capsys.readouterr().out.splitlines() == ["2/5 judged", "4/5 judged"]


def test_judge_rows_counts_failed_rows_in_the_progress(capsys):
    client = client_answering(CONNECTION_ERROR, verdict_response(EQUIVALENT))
    judge_rows(client, [RUSSELL_EVAL_ROW, LEVIS_EVAL_ROW], FACTS_BY_ID, progress_every=2)
    assert capsys.readouterr().out.splitlines() == ["2/2 judged"]


def test_judge_rows_prints_no_progress_unless_asked(capsys):
    client = client_answering(*[verdict_response(EQUIVALENT)] * 5)
    judge_rows(client, [RUSSELL_EVAL_ROW] * 5, FACTS_BY_ID)
    assert capsys.readouterr().out == ""


def test_judged_rows_of_are_the_rows_that_got_a_verdict():
    outcomes = [Outcome(RUSSELL_JUDGED_ROW, None), failed_outcome(LEVIS_EVAL_ROW, "stop_reason: refusal")]
    assert judged_rows_of(outcomes) == [RUSSELL_JUDGED_ROW]


def test_error_rows_of_are_the_rows_that_failed():
    outcomes = [Outcome(RUSSELL_JUDGED_ROW, None), failed_outcome(LEVIS_EVAL_ROW, "stop_reason: refusal")]
    assert error_rows_of(outcomes) == [{"listing_id": 8, "error": "stop_reason: refusal"}]


def test_judge_summary_is_each_check_rate_each_overall_share_and_the_row_count():
    assert judge_summary([RUSSELL_JUDGED_ROW, LEVIS_JUDGED_ROW]) == HALF_WRONG_SUMMARY


def test_judge_summary_gives_the_mean_similarity_and_the_share_scoring_four_or_five():
    rows = [{**RUSSELL_JUDGED_ROW, "similarity": similarity} for similarity in (5, 2, 4)]
    summary = judge_summary(rows)
    assert (round(summary["similarity_mean"], 2), round(summary["similarity_4_or_5"], 3)) == (3.67, 0.667)


def test_judge_summary_leaves_refused_and_failed_rows_out_of_the_rates():
    client = client_answering(verdict_response(EQUIVALENT), response_stopped_by("refusal"), CONNECTION_ERROR)
    outcomes = judge_rows(client, [RUSSELL_EVAL_ROW, LEVIS_EVAL_ROW, RUSSELL_EVAL_ROW], FACTS_BY_ID)
    summary = judge_summary(judged_rows_of(outcomes))
    assert (summary["rows"], summary["overall_equivalent"]) == (1, 1.0)


def test_judge_summary_raises_when_no_row_was_judged():
    with pytest.raises(ZeroDivisionError):
        judge_summary([])


def test_summary_line_shows_a_rate_as_a_percentage():
    assert summary_line("same_item", 0.5) == "same_item            50.0%"


def test_summary_line_shows_the_mean_similarity_out_of_five():
    assert summary_line("similarity_mean", 3.6667) == "similarity_mean      3.67 out of 5"


def test_summary_lines_are_the_rates_then_row_error_and_token_counts():
    outcomes = [
        Outcome(RUSSELL_JUDGED_ROW, None, input_tokens=400, output_tokens=60),
        Outcome(LEVIS_JUDGED_ROW, None, input_tokens=410, output_tokens=70),
        failed_outcome(EVAL_ROW_MISSING_FROM_VAL, "stop_reason: refusal", input_tokens=400, output_tokens=5),
    ]
    assert summary_lines(HALF_WRONG_SUMMARY, outcomes) == [
        "same_item            100.0%",
        "brand_agrees         100.0%",
        "era_agrees           100.0%",
        "nothing_invented     50.0%",
        "key_details_kept     50.0%",
        "similarity_mean      3.50 out of 5",
        "similarity_4_or_5    50.0%",
        "overall_equivalent   50.0%",
        "overall_acceptable   0.0%",
        "overall_wrong        50.0%",
        "rows judged          2",
        "rows with an error   1",
        "input tokens         1210",
        "output tokens        135",
    ]


def test_verdict_lines_show_the_shop_line_the_generated_line_the_verdict_and_the_reason():
    assert verdict_lines(LEVIS_JUDGED_ROW) == [
        "    8  shop:      💎 Vintage 90's Levi's Stonewash Jeans.",
        "       generated: 💎 Vintage 90's Levi's 550 Jeans.",
        "       verdict:   wrong, similarity 2/5 (failed: nothing_invented, key_details_kept)",
        "       reason:    The generated line adds a 550 fit number and drops the stonewash.",
        "",
    ]


def test_verdict_lines_say_none_failed_when_every_check_passed():
    assert verdict_lines(RUSSELL_JUDGED_ROW)[2] == "       verdict:   equivalent, similarity 5/5 (failed: none)"


def test_error_lines_are_one_not_judged_line_per_error_row():
    assert error_lines([{"listing_id": 8, "error": "stop_reason: refusal"}]) == [
        "    8  not judged: stop_reason: refusal"
    ]


def test_write_judge_file_writes_one_judged_row_per_line(judge_file):
    write_judge_file([RUSSELL_JUDGED_ROW, LEVIS_JUDGED_ROW])
    lines = judge_file.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [RUSSELL_JUDGED_ROW, LEVIS_JUDGED_ROW]


def test_write_judge_file_writes_non_ascii_characters_unescaped(judge_file):
    write_judge_file([RUSSELL_JUDGED_ROW])
    assert '"target": "💎 Vintage 90\'s Boxy Russell Hoodie."' in judge_file.read_text(encoding="utf-8")


def test_row_limit_from_is_the_first_argument_as_a_number():
    assert row_limit_from(["10"]) == 10


def test_row_limit_from_is_none_without_an_argument():
    assert row_limit_from([]) is None


def test_judge_pilot_judges_only_the_first_rows_up_to_the_limit(eval_and_val_files, capsys):
    client = client_answering(verdict_response(EQUIVALENT))
    judge_pilot(client, 1)
    assert len(client.messages.requests) == 1
    assert "rows judged          1" in capsys.readouterr().out


def test_judge_pilot_prints_each_pair_with_its_verdict_for_reading(eval_and_val_files, capsys):
    judge_pilot(client_answering(verdict_response(EQUIVALENT)), 1)
    printed = capsys.readouterr().out
    assert "\n".join(verdict_lines(RUSSELL_JUDGED_ROW)) in printed


def test_judge_pilot_prints_the_rows_that_were_not_judged(eval_and_val_files, capsys):
    judge_pilot(client_answering(verdict_response(EQUIVALENT), response_stopped_by("refusal")), 2)
    assert "    8  not judged: stop_reason: refusal" in capsys.readouterr().out


def test_judge_pilot_does_not_log_to_mlflow(eval_and_val_files, judge_run_log):
    judge_pilot(client_answering(verdict_response(EQUIVALENT)), 1)
    assert judge_run_log == []


def test_judge_pilot_does_not_write_the_judge_file(eval_and_val_files, judge_file):
    judge_pilot(client_answering(verdict_response(EQUIVALENT)), 1)
    assert not judge_file.exists()


def test_judge_pilot_prints_no_progress_lines(twenty_five_eval_rows, capsys):
    judge_pilot(client_answering(*twenty_five_eval_rows), 25)
    assert "20/25 judged" not in capsys.readouterr().out


def test_judge_everything_judges_every_row_of_the_eval_file(eval_and_val_files, judge_run_log):
    client = client_answering(verdict_response(EQUIVALENT), verdict_response(INVENTED_FIT_NUMBER))
    judge_everything(client)
    assert len(client.messages.requests) == 2


def test_judge_everything_writes_the_judged_rows_to_the_judge_file(eval_and_val_files, judge_file, judge_run_log):
    judge_everything(client_answering(verdict_response(EQUIVALENT), verdict_response(INVENTED_FIT_NUMBER)))
    lines = judge_file.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [RUSSELL_JUDGED_ROW, LEVIS_JUDGED_ROW]


def test_judge_everything_hands_rows_summary_model_rubric_and_file_to_the_run_log(
    eval_and_val_files, judge_file, judge_run_log
):
    judge_everything(client_answering(verdict_response(EQUIVALENT), verdict_response(INVENTED_FIT_NUMBER)))
    assert judge_run_log == [
        ([RUSSELL_JUDGED_ROW, LEVIS_JUDGED_ROW], HALF_WRONG_SUMMARY, "claude-haiku-4-5", RUBRIC, judge_file)
    ]


def test_judge_everything_prints_the_errors_and_the_summary(eval_and_val_files, judge_run_log, capsys):
    judge_everything(client_answering(verdict_response(EQUIVALENT), CONNECTION_ERROR))
    printed = capsys.readouterr().out
    assert "    8  not judged: APIConnectionError: Connection error." in printed
    assert "rows with an error   1" in printed


def test_judge_everything_raises_before_writing_or_logging_when_no_row_was_judged(
    eval_and_val_files, judge_file, judge_run_log
):
    with pytest.raises(ZeroDivisionError):
        judge_everything(client_answering(CONNECTION_ERROR, CONNECTION_ERROR))
    assert not judge_file.exists()
    assert judge_run_log == []


def test_judge_everything_prints_a_progress_line_every_twenty_rows(twenty_five_eval_rows, judge_run_log, capsys):
    judge_everything(client_answering(*twenty_five_eval_rows))
    assert capsys.readouterr().out.splitlines()[0] == "20/25 judged"


def test_judge_everything_stops_without_writing_or_logging_when_the_key_is_rejected(
    eval_and_val_files, judge_file, judge_run_log
):
    with pytest.raises(SystemExit) as stopped:
        judge_everything(client_answering(KEY_REJECTED_ERROR))
    assert stopped.value.code == KEY_REJECTED_MESSAGE
    assert not judge_file.exists()
    assert judge_run_log == []


def test_main_stops_with_a_clear_message_before_creating_a_client_when_the_key_is_missing(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY_IS_SET", False)
    monkeypatch.setattr(judge.anthropic, "Anthropic", lambda: pytest.fail("a client was created without a key"))
    with pytest.raises(SystemExit) as stopped:
        judge.main()
    assert stopped.value.code == "ANTHROPIC_API_KEY is missing or blank in .env, so no row was judged."


def test_main_judges_everything_with_one_client_when_given_no_argument(monkeypatch):
    clients_used = []
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY_IS_SET", True)
    monkeypatch.setattr(judge.anthropic, "Anthropic", lambda: "fake client")
    monkeypatch.setattr(judge, "judge_everything", clients_used.append)
    monkeypatch.setattr(sys, "argv", ["judge"])
    judge.main()
    assert clients_used == ["fake client"]


def test_main_runs_the_pilot_on_the_number_of_rows_given_as_the_argument(monkeypatch):
    pilots = []
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY_IS_SET", True)
    monkeypatch.setattr(judge.anthropic, "Anthropic", lambda: "fake client")
    monkeypatch.setattr(judge, "judge_pilot", lambda client, row_limit: pilots.append((client, row_limit)))
    monkeypatch.setattr(sys, "argv", ["judge", "10"])
    judge.main()
    assert pilots == [("fake client", 10)]
