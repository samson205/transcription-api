from api.engines.diarization_engine import DiarizationEngine


class EmbeddingService:
    def __init__(self, engine: DiarizationEngine) -> None:
        self._engine = engine

    def extract_embedding(self, file_path: str) -> list[float]:
        output = self._engine.diarize_audio(file_path)

        if not output:
            raise ValueError("Failed to process audiofile")

        embeddings_array = output[1]
        if embeddings_array is None or len(embeddings_array) == 0:
            raise ValueError(
                "Diarization pipeline failed to extract a single embedding"
            )

        raw_embedding = (
            embeddings_array[0] if embeddings_array.ndim > 1 else embeddings_array
        )
        return raw_embedding.tolist()
