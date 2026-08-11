import csv
from pathlib import Path

from api.schemas.metrics import FileMetric, SegmentMetric


class MetricsService:
    def __init__(self, path: str = "export/metrics.csv") -> None:
        self.path = Path(path)

    def _init_file(self, header_rows: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(
                    header_rows
                )


class FileMetricsService(MetricsService):
    def append(self, data: FileMetric) -> None:
        self._init_file(
            [
                "filename", "device", "compute_type",
                "whisper_model", "processing_time", "file_duration",
                "num_segments", "method", "best_operator",
                "best_score", "second_score", "margin", "error",
            ]
        )
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(
                [
                    data.filename,
                    data.device,
                    data.compute_type,
                    data.whisper_model,
                    data.processing_time,
                    data.file_duration,
                    data.num_segments,
                    data.method if data.method else "",
                    data.best_operator if data.best_operator else "",
                    round(data.best_score, 4) if data.best_score else "",
                    round(data.second_score, 4) if data.second_score else "",
                    round(data.margin, 4) if data.margin else "",
                    data.error if data.error else "",
                ]
            )


class SegmentMetricsService(MetricsService):
    def append(self, data: list[SegmentMetric]) -> None:
        self._init_file(
            [
                "filename", "start", "end",
                "duration", "text", "speaker",
                "distance", "source", "method",
                "best_operator",
            ]
        )
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            for row in data:
                writer.writerow(
                    [
                        row.filename,
                        round(row.start, 2),
                        round(row.end, 2),
                        round(row.duration, 2),
                        row.text,
                        row.speaker,
                        round(row.distance) if row.distance is not None else "",
                        row.source,
                        row.method,
                        row.best_operator or ""
                    ]
                )
                