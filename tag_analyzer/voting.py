from collections import defaultdict

from tag_analyzer.schema import Neighbor


def vote(neighbors: list[Neighbor]) -> tuple[str, str, float]:
    brand_weight = defaultdict(float)
    era_weight = defaultdict(float)

    for neighbor in neighbors:
        brand_weight[neighbor.brand] += neighbor.similarity
        era_weight[neighbor.era] += neighbor.similarity

    winning_brand = max(brand_weight, key=brand_weight.get)
    winning_era = max(era_weight, key=era_weight.get)
    winning_brand_share = brand_weight[winning_brand] / sum(brand_weight.values())

    return winning_brand, winning_era, winning_brand_share
