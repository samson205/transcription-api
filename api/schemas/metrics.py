from pydantic import BaseModel


class FileMetric(BaseModel):
    filename: str
    duration: float
    num_segments: int | None = None
    method: str | None = None
    best_operator: str | None = None
    best_score: float | None = None
    second_score: float | None = None
    margin: float | None = None
    error: str | None = None
