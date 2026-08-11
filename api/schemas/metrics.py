from pydantic import BaseModel


class FileMetric(BaseModel):
    filename: str
    device: str
    compute_type: str
    whisper_model: str
    processing_time: float | None = None
    file_duration: float | None = None
    num_segments: int | None = None
    method: str | None = None
    best_operator: str | None = None
    best_score: float | None = None
    second_score: float | None = None
    margin: float | None = None
    error: str | None = None


class SegmentMetric(BaseModel):
    filename: str
    start: float
    end: float
    duration: float
    text: str
    speaker: str
    distance: float | None = None
    source: str # откуда взялось решение
    method: str
    best_operator: str | None = None
