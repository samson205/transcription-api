import logging

import numpy as np
from pyannote.core import Segment
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering

from api.core.config import settings
from api.services.operator_service import OperatorService
from api.services.embedding_service import EmbeddingService
from api.services.metrics_service import MetricsCSVService
from api.schemas.transcription import DialogueSegment
from api.schemas.metrics import FileMetric

logger = logging.getLogger(__name__)


class SpeakerMatchService:
    _MAX_CANDIDATES: int = 20

    _MIN_SEGMENT_DURATION = 2.0
    _MAX_SEGMENT_DURATION = 10.0

    _MIN_CLUSTER_SHARE = 0.15
    _MIN_CLUSTER_MARGIN = 0.04

    _MIN_SEGMENTS_FOR_CLUSTERING = 12

    def __init__(
        self, operator_service: OperatorService, embedding_service: EmbeddingService
    ) -> None:
        self._operator_service = operator_service
        self._embedding_service = embedding_service
        self._metrics_service = MetricsCSVService()

    async def match_operators(
        self, segments: list[DialogueSegment], path: str, original_filename: str
    ) -> tuple[list[DialogueSegment], int | None]:
        audio_in_memory = self._embedding_service.load_audio(path)

        embeddings = {}
        for segment in segments:
            excerpt = Segment(segment.start, segment.end)
            key = (segment.start, segment.end)
            try:
                embeddings[key] = self._embedding_service.extract_embedding(
                    audio_in_memory, excerpt
                )
            except Exception:
                logger.exception("Failed to extract embedding for segment")
                embeddings[key] = None

        total_duration = sum(s.end - s.start for s in segments)
        metric = FileMetric(filename=original_filename, duration=total_duration)
        best_operator = await self._identify_operator(
            segments, embeddings, original_filename, metric
        )
        if not best_operator:
            if settings.DEBUG_METRICS:
                self._metrics_service.append(metric)
            return segments, None

        target_operator_vector = best_operator.embedding
        if not target_operator_vector:
            if settings.DEBUG_METRICS:
                self._metrics_service.append(metric)
            return segments, None

        matched_segments = []
        for segment in segments:
            duration = segment.end - segment.start
            resolved_role = None
            if duration < 0.3:
                resolved_role = "Неизвестный"
            else:
                key = (segment.start, segment.end)
                segment_emb = embeddings[key]
                if segment_emb is None:
                    resolved_role = "Неизвестный"
                    upd_segment = segment.model_copy(update={"speaker": resolved_role})
                    matched_segments.append(upd_segment)
                    continue

                dist_to_operator = cosine(segment_emb, target_operator_vector)
                if dist_to_operator <= settings.THRESHOLD:
                    resolved_role = f"Оператор ({best_operator.name})"
                elif dist_to_operator <= settings.UNCERTAIN_BOUND:
                    resolved_role = f"Оператор ({best_operator.name}) [Неуверенно]"
                else:
                    resolved_role = "Клиент"

            upd_segment = segment.model_copy(update={"speaker": resolved_role})
            matched_segments.append(upd_segment)

        if settings.DEBUG_METRICS:
            self._metrics_service.append(metric)
        return matched_segments, best_operator.id

    async def _identify_operator(
        self,
        segments: list[DialogueSegment],
        embeddings: dict,
        original_filename: str,
        metric: FileMetric,
    ):
        total_duration = sum(s.end - s.start for s in segments)
        if total_duration <= 60 or len(segments) < 12:
            metric.method = "simple"
            operator = await self._identify_operator_simple(
                segments, embeddings, original_filename, metric
            )

        elif total_duration <= 180 or len(segments) <= 20:
            metric.method = "cluster"
            operator = await self._identify_operator_by_cluster(
                segments, embeddings, original_filename, metric
            )
            if operator is None:
                metric.method = "simple"
                operator = await self._identify_operator_simple(
                    segments, embeddings, original_filename, metric
                )

        else:
            metric.method = "cluster"
            operator = await self._identify_operator_by_cluster(
                segments, embeddings, original_filename, metric
            )

        if operator is None:
            logger.warning("Failed to identify operator file=%s", original_filename)
        return operator

    async def _identify_operator_simple(
        self,
        segments: list[DialogueSegment],
        embeddings: dict,
        original_filename: str,
        metric: FileMetric,
    ):
        candidates = [
            s
            for s in segments
            if self._MIN_SEGMENT_DURATION
            <= s.end - s.start
            <= self._MAX_SEGMENT_DURATION
        ]
        metric.num_segments = len(candidates)
        if len(candidates) <= self._MAX_CANDIDATES:
            chunks_to_analyze = candidates
        else:
            step = max(1, len(candidates) // self._MAX_CANDIDATES)
            chunks_to_analyze = candidates[::step][: self._MAX_CANDIDATES]

        num_chunks = len(chunks_to_analyze)
        if num_chunks < 10:
            min_margin = 0.03
        elif num_chunks < 20:
            min_margin = 0.04
        else:
            min_margin = 0.05

        operators = {}
        operator_distances = {}

        for segment in chunks_to_analyze:
            key = (segment.start, segment.end)
            segment_emb = embeddings[key]
            if segment_emb is None:
                continue

            operator, distance = await self._operator_service.find_matching_operator(
                segment_emb
            )
            if not operator:
                continue

            if distance <= settings.THRESHOLD:
                operator_distances.setdefault(operator.id, []).append(distance)
                operators[operator.id] = operator

        scores = []
        for op_id, dists in operator_distances.items():
            # score = np.percentile(dists, 25)
            sorted_dists = sorted(dists)
            top_dists = sorted_dists[: min(2, len(sorted_dists))]
            score = np.mean(top_dists)
            logger.info("operator=%s score=%s", operators[op_id].name, str(score))
            scores.append(
                (
                    score,
                    op_id,
                )
            )

        if not scores:
            metric.error = "no_scores"
            return None

        scores.sort()
        best_score, best_id = scores[0]
        metric.best_score = best_score
        metric.best_operator = operators[best_id].name
        if best_score > settings.THRESHOLD:
            metric.error = "best_score > threshold"
            return None

        if len(scores) > 1:
            second_score, _ = scores[1]
            metric.second_score = second_score
            metric.margin = second_score - best_score
            if second_score - best_score < min_margin:
                metric.error = "margin < min_margin"
                return None

        logger.info("Operator found by simple strategy file=%s", original_filename)
        return operators[best_id]

    async def _identify_operator_by_cluster(
        self,
        segments: list[DialogueSegment],
        embeddings: dict,
        original_filename: str,
        metric: FileMetric,
    ):
        candidates = [
            s
            for s in segments
            if self._MIN_SEGMENT_DURATION
            <= s.end - s.start
            <= self._MAX_SEGMENT_DURATION
        ]
        metric.num_segments = len(candidates)
        clusters = self._cluster_embeddings(
            candidates,
            embeddings,
        )

        if clusters is None:
            metric.error = "no clusters"
            return None

        cluster_scores = []
        total_duration = sum(s.end - s.start for s in segments)
        for cluster_id, cluster_segments in clusters.items():
            centroid = self._cluster_centroid(cluster_segments, embeddings)
            operator, distance = await self._operator_service.find_matching_operator(
                centroid
            )
            if not operator:
                continue

            cluster_duration = sum(s.end - s.start for s in cluster_segments)
            share = cluster_duration / total_duration
            if share < self._MIN_CLUSTER_SHARE:
                continue

            if len(cluster_segments) < 3:
                continue

            logger.info(
                "cluster=%s operator=%s distance=%.3f duration=%.1f",
                cluster_id,
                operator.name,
                distance,
                cluster_duration,
            )

            cluster_scores.append(
                (
                    distance,
                    operator,
                )
            )

        if not cluster_scores:
            metric.error = "no cluster scores"
            return None

        cluster_scores.sort(key=lambda x: x[0])

        best_distance, best_operator = cluster_scores[0]
        metric.best_score = best_distance
        metric.best_operator = best_operator.name
        if len(cluster_scores) > 1:
            second_distance, _ = cluster_scores[1]
            metric.second_score = second_distance
            metric.margin = second_distance - best_distance

            if second_distance - best_distance < self._MIN_CLUSTER_MARGIN:
                metric.error = "margin < min_margin"
                logger.info(
                    "Clusters are too close: %.3f vs %.3f",
                    best_distance,
                    second_distance,
                )
                return None

        if best_distance > settings.THRESHOLD:
            metric.error = "best_score < threshold"
            return None

        logger.info("Operator found by cluster strategy file=%s", original_filename)
        return best_operator

    def _cluster_embeddings(self, candidates: list, embeddings: dict):
        valid_candidates = [
            s for s in candidates if embeddings[(s.start, s.end)] is not None
        ]

        x = np.stack([embeddings[(s.start, s.end)] for s in valid_candidates])

        clusterer = AgglomerativeClustering(
            n_clusters=2, metric="cosine", linkage="average"
        )

        labels = clusterer.fit_predict(x)

        clusters = {}
        for segment, label in zip(valid_candidates, labels):
            clusters.setdefault(int(label), []).append(segment)
        return clusters

    def _cluster_centroid(self, cluster: list[DialogueSegment], embeddings: dict):
        vectors = [embeddings[(s.start, s.end)] for s in cluster]

        centroid = np.mean(vectors, axis=0)
        centroid /= np.linalg.norm(centroid)
        return centroid.tolist()
