import pytest

from tag_analyzer.schema import Neighbor
from tag_analyzer.voting import vote


def neighbor(brand: str, era: str, similarity: float) -> Neighbor:
    return Neighbor(filename=f"{brand}/{era}.jpg", brand=brand, era=era, similarity=similarity)


def test_vote_picks_the_brand_and_era_of_a_single_neighbor():
    brand, era, _ = vote([neighbor("Russell", "90s", 0.9)])
    assert (brand, era) == ("Russell", "90s")


def test_vote_weights_each_neighbor_by_similarity():
    brand, _, _ = vote(
        [neighbor("Russell", "90s", 0.9), neighbor("Champion", "90s", 0.4), neighbor("Champion", "90s", 0.4)]
    )
    assert brand == "Russell"


def test_vote_adds_up_the_similarity_of_neighbors_with_the_same_brand():
    brand, _, _ = vote(
        [neighbor("Russell", "90s", 0.9), neighbor("Champion", "90s", 0.5), neighbor("Champion", "90s", 0.5)]
    )
    assert brand == "Champion"


def test_vote_confidence_is_the_winning_brands_share_of_the_total_similarity():
    _, _, confidence = vote([neighbor("Russell", "90s", 0.75), neighbor("Champion", "90s", 0.25)])
    assert confidence == pytest.approx(0.75)


def test_vote_confidence_is_one_when_every_neighbor_agrees_on_the_brand():
    _, _, confidence = vote([neighbor("Russell", "90s", 0.9), neighbor("Russell", "80s", 0.8)])
    assert confidence == pytest.approx(1.0)


def test_vote_decides_brand_and_era_independently():
    brand, era, _ = vote(
        [neighbor("Russell", "90s", 0.6), neighbor("Champion", "80s", 0.35), neighbor("Champion", "70s", 0.35)]
    )
    assert (brand, era) == ("Champion", "90s")


def test_vote_raises_on_an_empty_neighbor_list():
    with pytest.raises(ValueError):
        vote([])
