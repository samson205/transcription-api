import csv
from pathlib import Path

from api.schemas.metrics import FileMetric


class MetricsCSVService:
    def __init__(self, path: str = "export/metrics.csv") -> None:
        self.path = Path(path)

    def _init_file(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow([
                    "filename",
                    "duration",
                    "num_segments",
                    "method",
                    "best_operator",
                    "best_score",
                    "second_score",
                    "margin",
                    "error",
                ])

    def append(self, data: FileMetric):
        self._init_file()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                data.filename,
                data.duration,
                data.num_segments,
                data.method if data.method else "",
                data.best_operator if data.best_operator else "",
                round(data.best_score, 4) if data.best_score else "",
                round(data.second_score, 4) if data.second_score else "",
                round(data.margin, 4) if data.margin else "",
                data.error if data.error else "",
            ])
