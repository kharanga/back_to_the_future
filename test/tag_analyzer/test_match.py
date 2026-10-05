import pytest

from tag_analyzer import store
from tag_analyzer.match import K, THRESHOLD, match
from tag_analyzer.schema import Neighbor

ABOVE_THRESHOLD = THRESHOLD + 0.1
BELOW_THRESHOLD = THRESHOLD - 0.1


def neighbor(brand: str, era: str, similarity: float) -> Neighbor:
    return Neighbor(filename=f"{brand}/{era}.jpg", brand=brand, era=era, similarity=similarity)


def search_returns(monkeypatch, neighbors: list[Neighbor]):
    monkeypatch.setattr(store, "search_nearest_photos", lambda client, query_photo, k: neighbors)


def test_match_is_not_in_library_when_the_search_finds_no_neighbors(monkeypatch):
    search_returns(monkeypatch, [])
    result = match("fake client", "tag.jpg")
    assert (result.in_library, result.brand, result.era, result.confidence) == (False, None, None, 0.0)


def test_match_is_not_in_library_when_the_top_similarity_is_below_the_threshold(monkeypatch):
    search_returns(monkeypatch, [neighbor("Russell", "90s", BELOW_THRESHOLD)])
    result = match("fake client", "tag.jpg")
    assert (result.in_library, result.brand, result.era, result.confidence) == (False, None, None, 0.0)


def test_match_keeps_the_neighbors_when_it_is_not_in_library(monkeypatch):
    search_returns(monkeypatch, [neighbor("Russell", "90s", BELOW_THRESHOLD)])
    assert match("fake client", "tag.jpg").neighbors == [neighbor("Russell", "90s", BELOW_THRESHOLD)]


def test_match_is_in_library_when_the_top_similarity_equals_the_threshold(monkeypatch):
    search_returns(monkeypatch, [neighbor("Russell", "90s", THRESHOLD)])
    assert match("fake client", "tag.jpg").in_library


def test_match_takes_brand_and_era_from_the_vote_when_in_library(monkeypatch):
    search_returns(
        monkeypatch,
        [neighbor("Russell", "90s", ABOVE_THRESHOLD), neighbor("Russell", "90s", 0.5), neighbor("Champion", "80s", 0.5)],
    )
    result = match("fake client", "tag.jpg")
    assert (result.in_library, result.brand, result.era) == (True, "Russell", "90s")


def test_match_confidence_is_the_winning_brands_share_of_the_vote(monkeypatch):
    search_returns(monkeypatch, [neighbor("Russell", "90s", 0.75), neighbor("Champion", "80s", 0.25)])
    assert match("fake client", "tag.jpg").confidence == pytest.approx(0.75)


def test_match_searches_for_k_neighbors_of_the_query_photo(monkeypatch):
    search_arguments = []
    monkeypatch.setattr(
        store,
        "search_nearest_photos",
        lambda client, query_photo, k: search_arguments.append((client, query_photo, k)) or [],
    )
    match("fake client", "tag.jpg")
    assert search_arguments == [("fake client", "tag.jpg", K)]
