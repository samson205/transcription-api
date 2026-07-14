# """
# Тесты для SpeakerMatchService.match_operators

# Это самая важная бизнес-логика проекта: она решает, кто из двух спикеров —
# оператор, а кто клиент, и насколько уверенно. OperatorService мокается
# через AsyncMock, чтобы не поднимать реальную БД
# """

# from types import SimpleNamespace
# from unittest.mock import AsyncMock

# from api.services.speaker_match_service import SpeakerMatchService
# from api.schemas.transcription import DialogueSegment


# def make_operator(name: str) -> SimpleNamespace:
#     return SimpleNamespace(name=name)


# def make_segment(speaker: str, text: str = "привет") -> DialogueSegment:
#     return DialogueSegment(start=0.0, end=1.0, speaker=speaker, text=text)


# class TestSpeakerMatchService:
#     def setup_method(self) -> None:
#         self.operator_service = AsyncMock()
#         self.service = SpeakerMatchService(self.operator_service)

#     async def test_two_speakers_one_confident_operator_match(self) -> None:
#         # SPEAKER_00 уверенно совпал с оператором (distance < threshold),
#         # SPEAKER_01 совпадения не имеет -> помечается как клиент
#         self.operator_service.find_matching_operator.side_effect = [
#             (make_operator("Иван"), 0.2),
#             (None, 1.0),
#         ]
#         segments = [make_segment("SPEAKER_00"), make_segment("SPEAKER_01")]
#         embeddings = {"SPEAKER_00": [0.1, 0.2], "SPEAKER_01": [0.5, 0.6]}

#         result = await self.service.match_operators(segments, embeddings)

#         assert result[0].speaker == "Оператор (Иван)"
#         assert result[1].speaker == "Клиент"

#     async def test_two_speakers_both_below_confidence_threshold(self) -> None:
#         # Обе дистанции выше threshold (0.4), но одна из них ниже
#         # absolte_max_dist (0.6) -> "неуверенное" совпадение
#         self.operator_service.find_matching_operator.side_effect = [
#             (make_operator("Иван"), 0.55),
#             (None, 0.9),
#         ]
#         segments = [make_segment("SPEAKER_00"), make_segment("SPEAKER_01")]
#         embeddings = {"SPEAKER_00": [0.1], "SPEAKER_01": [0.2]}

#         result = await self.service.match_operators(segments, embeddings)

#         assert result[0].speaker == "Оператор (Иван) [Неуверенно]"
#         assert result[1].speaker == "Клиент"

#     async def test_two_speakers_neither_matches_any_operator(self) -> None:
#         # Обе дистанции превышают даже absolte_max_dist -> оба помечаются
#         # как неопределённые, без привязки к конкретному оператору
#         self.operator_service.find_matching_operator.side_effect = [
#             (None, 0.8),
#             (None, 0.9),
#         ]
#         segments = [make_segment("SPEAKER_00"), make_segment("SPEAKER_01")]
#         embeddings = {"SPEAKER_00": [0.1], "SPEAKER_01": [0.2]}

#         result = await self.service.match_operators(segments, embeddings)

#         assert result[0].speaker == "Оператор [Неизвестен / Неуверенно]"
#         assert result[1].speaker == "Клиент [Неуверенно]"

#     async def test_single_speaker_confident_match(self) -> None:
#         self.operator_service.find_matching_operator.return_value = (
#             make_operator("Мария"),
#             0.1,
#         )
#         segments = [make_segment("SPEAKER_00")]
#         embeddings = {"SPEAKER_00": [0.1]}

#         result = await self.service.match_operators(segments, embeddings)

#         assert result[0].speaker == "Оператор (Мария)"

#     async def test_single_speaker_no_match_defaults_to_client(self) -> None:
#         self.operator_service.find_matching_operator.return_value = (None, 1.0)
#         segments = [make_segment("SPEAKER_00")]
#         embeddings = {"SPEAKER_00": [0.1]}

#         result = await self.service.match_operators(segments, embeddings)

#         assert result[0].speaker == "Клиент / Неизвестно"

#     async def test_segment_with_unmapped_speaker_id_falls_back(self) -> None:
#         # Сегмент ссылается на спикера, которого нет среди embeddings
#         # (например, диаризация и транскрипция разошлись) -> не должно падать
#         self.operator_service.find_matching_operator.return_value = (None, 1.0)
#         segments = [make_segment("SPEAKER_99")]
#         embeddings = {"SPEAKER_00": [0.1]}

#         result = await self.service.match_operators(segments, embeddings)

#         assert result[0].speaker == "Неизвестный"

#     async def test_accepts_numpy_like_embeddings_with_tolist(self) -> None:
#         # DiarizationEngine отдаёт эмбеддинги как numpy-массивы (.tolist())
#         numpy_like = SimpleNamespace(tolist=lambda: [0.1, 0.2, 0.3])
#         self.operator_service.find_matching_operator.return_value = (
#             make_operator("Иван"),
#             0.1,
#         )
#         segments = [make_segment("SPEAKER_00")]
#         embeddings = {"SPEAKER_00": numpy_like}

#         result = await self.service.match_operators(segments, embeddings)

#         self.operator_service.find_matching_operator.assert_awaited_once_with(
#             [0.1, 0.2, 0.3]
#         )
#         assert result[0].speaker == "Оператор (Иван)"
