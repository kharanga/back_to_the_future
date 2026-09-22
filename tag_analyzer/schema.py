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

