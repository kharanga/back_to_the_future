from pydantic import BaseModel

class Neighbor(BaseModel):
    filename: str
    brand: str
    era: str
    similarity: float

class TagMatch(BaseModel):
    brand: str | None = None
    era: str | None = None
    confidence: float
    neighbors: list[Neighbor]
    in_library: bool

class EvalScore(BaseModel):
    test_set_size: int
    correct_brand: int
    correct_era: int
    mistakes: list[str]

