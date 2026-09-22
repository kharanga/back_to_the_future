from tag_analyzer.schema import Neighbor
from collections import defaultdict
import numpy as np

def top_k(sims: np.ndarray, labels: list[dict], k: int) -> list[Neighbor]:    
    top_idx = np.argsort(sims)[-k:][::-1]
    return [
        Neighbor(
            filename=labels[i]["filename"],
            brand=labels[i]["brand"],
            era=labels[i]["era"],
            similarity=float(sims[i]),
        )
        for i in top_idx
    ]
def vote(neighbors: list[Neighbor]) -> tuple[str, str, float]:
    brand_weight = defaultdict(float)
    era_weight = defaultdict(float)

    for n in neighbors:
        brand_weight[n.brand] += n.similarity
        era_weight[n.era]     += n.similarity

    winning_brand = max(brand_weight, key=brand_weight.get)
    winning_era = max(era_weight, key=era_weight.get)

    confidence = brand_weight[winning_brand] / sum(brand_weight.values())

    return winning_brand, winning_era, confidence

if __name__ == "__main__":
    fake_sims = np.array([0.91, 0.74, 0.88, 0.69, 0.81])
    fake_labels = [
        {"filename": "r70_1.jpg", "brand": "russell athletic", "era": "70s"},
        {"filename": "champ.jpg", "brand": "champion", "era": "80s"},
        {"filename": "r70_2.jpg", "brand": "russell athletic", "era": "70s"},
        {"filename": "fotl.jpg", "brand": "fruit of the loom", "era": "90s"},
        {"filename": "r90.jpg", "brand": "russell athletic", "era": "90s"},
    ]

    neighbors = top_k(fake_sims, fake_labels, k=4)
    print([f"{n.filename} {n.similarity:.2f}" for n in neighbors])

    brand, era, confidence = vote(neighbors)
    print(f"brand: {brand} | era: {era} | confidence: {confidence:.2f}")