import logging

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering

from api.core.config import settings
from api.models.operator_model import Operator
from api.services.operator_service import OperatorService
from api.services.embedding_service import EmbeddingService
from api.services.metrics_service import SegmentMetricsService
from api.schemas.transcription import DialogueSegment
from api.schemas.metrics import FileMetric, SegmentMetric

logger = logging.getLogger(__name__)


class SpeakerMatchService:
    _MAX_CANDIDATES: int = 20

    _MIN_SEGMENT_DURATION = 2.0
    _MAX_SEGMENT_DURATION = 15.0

    _MIN_CLUSTER_SHARE = 0.15
    _MIN_CLUSTER_MARGIN = 0.04
    _MIN_SEGMENT_FOR_CLUSTERING = 10

    def __init__(
        self, operator_service: OperatorService, embedding_service: EmbeddingService
    ) -> None:
        self._operator_service = operator_service
        self._embedding_service = embedding_service
        self._metrics_service = SegmentMetricsService("export/segment_metrics.csv")

    def assign_roles(
        self,
        segments: list[DialogueSegment],
        embeddings: dict,
        operator: Operator | None,
        original_filename: str,
        cluster_map: dict | None = None,
    ) -> list[DialogueSegment]:
        if operator is None:
            return segments
        matched_segments = []
        segment_metrics = []
        operator_embeddings = [oe.embedding for oe in operator.embeddings]
        for segment in segments:
            duration = segment.end - segment.start
            key = (segment.start, segment.end)
            distance = None
            resolved_role = None
            source = "skipped"

            if duration < 0.3:
                resolved_role = "Неизвестный"
            elif cluster_map and key in cluster_map:
                segment_emb = embeddings.get(key)
                if segment_emb is None:
                    resolved_role = "Неизвестный"
                    source = "skipped"
                else:
                    distance = self._distance_to_operator(
                        segment_emb, operator_embeddings
                    )
                    role = cluster_map[key]
                    if distance <= settings.UNCERTAIN_BOUND:
                        resolved_role = (
                            f"Оператор ({operator.name})"
                            if role == "operator"
                            else "Клиент"
                        )
                        source = "cluster"
                    else:
                        resolved_role = "Клиент"
                        source = "cluster_overriden"
            else:
                segment_emb = embeddings[key]
                if segment_emb is None:
                    resolved_role = "Неизвестный"
                    source = "skipped"
                else:
                    distance = self._distance_to_operator(
                        segment_emb, operator_embeddings
                    )
                    source = "cosine"
                    if distance <= settings.THRESHOLD:
                        resolved_role = f"Оператор ({operator.name})"
                    elif distance <= settings.UNCERTAIN_BOUND:
                        resolved_role = f"Оператор ({operator.name}) [Неуверенно]"
                    else:
                        resolved_role = "Клиент"

            segment_metrics.append(
                SegmentMetric(
                    filename=original_filename,
                    config=f"{settings.DEVICE}_{settings.COMPUTE_TYPE}_beam{settings.BEAM_SIZE}_wt{settings.WORD_TIMESTAMPS}",
                    start=segment.start,
                    end=segment.end,
                    duration=duration,
                    text=segment.text,
                    speaker=resolved_role,
                    distance=distance,
                    source=source,
                    best_operator=operator.name,
                )
            )

            upd_segment = segment.model_copy(update={"speaker": resolved_role})
            matched_segments.append(upd_segment)

        if settings.DEBUG_METRICS:
            self._metrics_service.append(segment_metrics)

        return matched_segments

    async def verify_claimed_operator(self, segments: list[DialogueSegment], embeddings: dict, claimed_operator: Operator) -> bool:
        candidates = self._get_valid_candidates(segments, embeddings)
        operator_embeddings = [oe.embedding for oe in claimed_operator.embeddings]
        matches = 0

        for segment in candidates:
            emb = embeddings.get((segment.start, segment.end))
            if emb is None:
                continue
            distance = self._distance_to_operator(emb, operator_embeddings)
            if distance <= settings.THRESHOLD:
                matches += 1

        return matches >= 2

    async def identify_operator(
        self,
        segments: list[DialogueSegment],
        embeddings: dict,
        original_filename: str,
        metric: FileMetric,
    ):
        candidates = self._get_valid_candidates(segments, embeddings)
        metric.num_segments = len(candidates)
        is_short_call = len(candidates) < self._MIN_SEGMENT_FOR_CLUSTERING
        if is_short_call:
            metric.method = "simple"
            operator = await self._identify_operator_simple(
                candidates, embeddings, original_filename, metric
            )
            return operator, None

        metric.method = "cluster"
        operator, cluster_map = await self._identify_operator_by_cluster(
            segments, embeddings, original_filename, metric
        )
        if operator is None:
            metric.method = "simple_fallback"
            metric.error = None
            metric.second_score = None
            metric.margin = None
            cluster_map = None
            operator = await self._identify_operator_simple(
                segments, embeddings, original_filename, metric
            )

        if operator is None:
            logger.warning("Failed to identify operator file=%s", original_filename)
        return operator, cluster_map

    async def _identify_operator_simple(
        self,
        candidates: list[DialogueSegment],
        embeddings: dict,
        original_filename: str,
        metric: FileMetric,
    ):
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
        candidates: list[DialogueSegment],
        embeddings: dict,
        original_filename: str,
        metric: FileMetric,
    ):
        clusters = self._cluster_embeddings(
            candidates,
            embeddings,
        )

        if clusters is None:
            metric.error = "no clusters"
            return None, None

        cluster_scores = []
        for cluster_id, cluster_segments in clusters.items():
            centroid = self._cluster_centroid(cluster_segments, embeddings)
            operator, distance = await self._operator_service.find_matching_operator(
                centroid
            )
            if not operator:
                continue

            if len(cluster_segments) < 3:
                continue

            cluster_duration = sum(s.end - s.start for s in cluster_segments)
            logger.info(
                "cluster=%s operator=%s distance=%.3f duration=%.1f",
                cluster_id,
                operator.name,
                distance,
                cluster_duration,
            )

            cluster_scores.append((distance, operator, cluster_id))

        if not cluster_scores:
            metric.error = "no cluster scores"
            return None, None

        cluster_scores.sort(key=lambda x: x[0])

        best_distance, best_operator, best_cluster_id = cluster_scores[0]
        metric.best_score = best_distance
        metric.best_operator = best_operator.name
        if len(cluster_scores) > 1:
            second_distance, _, _ = cluster_scores[1]
            metric.second_score = second_distance
            metric.margin = second_distance - best_distance

            if second_distance - best_distance < self._MIN_CLUSTER_MARGIN:
                metric.error = "margin < min_margin"
                logger.info(
                    "Clusters are too close: %.3f vs %.3f",
                    best_distance,
                    second_distance,
                )
                return None, None

        if best_distance > settings.THRESHOLD:
            metric.error = "best_score > threshold"
            return None, None

        logger.info("Operator found by cluster strategy file=%s", original_filename)

        cluster_map: dict[tuple[float, float], str] = {}
        for cluster_id, cluster_segments in clusters.items():
            role = "operator" if cluster_id == best_cluster_id else "client"
            for s in cluster_segments:
                cluster_map[(s.start, s.end)] = role

        return best_operator, cluster_map

    def _cluster_embeddings(self, candidates: list, embeddings: dict):
        if len(candidates) < 2:
            return None

        x = np.stack([embeddings[(s.start, s.end)] for s in candidates])

        clusterer = AgglomerativeClustering(
            n_clusters=2, metric="cosine", linkage="average"
        )

        labels = clusterer.fit_predict(x)

        clusters = {}
        for segment, label in zip(candidates, labels):
            clusters.setdefault(int(label), []).append(segment)
        return clusters

    def _cluster_centroid(self, cluster: list[DialogueSegment], embeddings: dict):
        vectors = [embeddings[(s.start, s.end)] for s in cluster]

        centroid = np.mean(vectors, axis=0)
        centroid /= np.linalg.norm(centroid)
        return centroid.tolist()

    def _get_valid_candidates(
        self, segments: list[DialogueSegment], embeddings: dict
    ) -> list[DialogueSegment]:
        candidates = [
            s
            for s in segments
            if self._MIN_SEGMENT_DURATION
            <= s.end - s.start
            <= self._MAX_SEGMENT_DURATION
        ]
        return [s for s in candidates if embeddings[(s.start, s.end)] is not None]

    def _distance_to_operator(
        self, segment_emb: list[float], operator_embeddings: list[list[float]]
    ) -> float:
        return min(cosine(segment_emb, ref) for ref in operator_embeddings)
