"""
Тесты для AlignmentService

Это чистая функция без внешних зависимостей (БД, ML-модели), но именно
она решает, какому спикеру достанется каждый кусок текста
"""

from api.services.alignment_service import AlignmentService
from api.schemas.transcription import SegmentSchema, SpeakerSegment


def make_segment(start: float, end: float, text: str = "текст") -> SegmentSchema:
    return SegmentSchema(start=start, end=end, text=text)


def make_speaker(start: float, end: float, speaker: str) -> SpeakerSegment:
    return SpeakerSegment(start=start, end=end, speaker=speaker)


class TestAlignmentService:
    def setup_method(self) -> None:
        self.service = AlignmentService()

    def test_picks_speaker_with_larger_overlap(self) -> None:
        transcription = [make_segment(0.0, 3.0)]
        diarization = [
            make_speaker(0.0, 0.5, "SPEAKER_00"),  # пересечение 0.5с
            make_speaker(0.5, 3.0, "SPEAKER_01"),  # пересечение 2.5с
        ]

        result = self.service.align(transcription, diarization)

        assert result[0].speaker == "SPEAKER_01"

    def test_equal_overlap_picks_last_candidate(self) -> None:
        transcription = [make_segment(0.0, 2.0)]
        diarization = [
            make_speaker(0.0, 1.0, "SPEAKER_00"),  # пересечение 1.0с
            make_speaker(1.0, 2.0, "SPEAKER_01"),  # пересечение 1.0с
        ]

        result = self.service.align(transcription, diarization)

        assert result[0].speaker == "SPEAKER_01"

    def test_no_overlap_returns_unknown_speaker(self) -> None:
        transcription = [make_segment(10.0, 11.0)]
        diarization = [make_speaker(0.0, 1.0, "SPEAKER_00")]

        result = self.service.align(transcription, diarization)

        assert result[0].speaker == "Unknown speaker"

    def test_empty_diarization_returns_unknown_speaker(self) -> None:
        transcription = [make_segment(0.0, 1.0)]

        result = self.service.align(transcription, [])

        assert result[0].speaker == "Unknown speaker"

    def test_preserves_order_timings_and_text(self) -> None:
        transcription = [
            make_segment(0.0, 1.0, "первый"),
            make_segment(1.0, 2.0, "второй"),
        ]
        diarization = [make_speaker(0.0, 2.0, "SPEAKER_00")]

        result = self.service.align(transcription, diarization)

        assert [s.text for s in result] == ["первый", "второй"]
        assert result[0].start == 0.0 and result[0].end == 1.0
        assert result[1].start == 1.0 and result[1].end == 2.0

    def test_multiple_speakers_multiple_segments(self) -> None:
        transcription = [
            make_segment(0.0, 1.0, "здравствуйте"),
            make_segment(1.0, 2.0, "здравствуйте, чем помочь"),
        ]
        diarization = [
            make_speaker(0.0, 1.0, "SPEAKER_00"),
            make_speaker(1.0, 2.0, "SPEAKER_01"),
        ]

        result = self.service.align(transcription, diarization)

        assert result[0].speaker == "SPEAKER_00"
        assert result[1].speaker == "SPEAKER_01"
    
    def test_multiple_segments_one_speaker(self) -> None:
        transcription = [make_segment(0.0, 1.0), make_segment(1.0, 2.0)]
        diarization = [make_speaker(0.0, 2.0, "SPEAKER_00")]

        result = self.service.align(transcription, diarization)

        assert result[0].speaker == "SPEAKER_00"
        assert result[1].speaker == "SPEAKER_00"
