"""
Тесты для SegmentAggregator.merge_by_sentences

Whisper отдаёт мелкие сегменты, и эта функция склеивает их в 
предложения по знакам препинания. Используем SimpleNamespace вместо
реальных Segment/Word из faster_whisper — merge_by_sentences работает
с любым объектом, у которого есть .start/.end и .text ИЛИ .word (duck
typing через getattr), так что не нужно тянуть настоящую модель ради теста
"""

from types import SimpleNamespace

from api.processors.segment_aggregator import SegmentAggregator


def raw_segment(start: float, end: float, text: str) -> SimpleNamespace:
    return SimpleNamespace(start=start, end=end, text=text)


class TestSegmentAggregator:
    def setup_method(self) -> None:
        self.aggregator = SegmentAggregator()

    def test_merges_words_until_sentence_end(self) -> None:
        segments = [
            raw_segment(0.0, 0.5, "Здравствуйте,"),
            raw_segment(0.5, 1.0, "чем"),
            raw_segment(1.0, 1.5, "могу"),
            raw_segment(1.5, 2.0, "помочь?"),
        ]

        result = self.aggregator.merge_by_sentences(segments) # type: ignore

        assert len(result) == 1
        assert result[0].text == "Здравствуйте, чем могу помочь?"
        assert result[0].start == 0.0
        assert result[0].end == 2.0

    def test_splits_into_separate_sentences(self) -> None:
        segments = [
            raw_segment(0.0, 1.0, "Привет."),
            raw_segment(1.0, 2.0, "Как дела?"),
        ]

        result = self.aggregator.merge_by_sentences(segments) # type: ignore

        assert [s.text for s in result] == ["Привет.", "Как дела?"]

    def test_handles_exclamation_and_question_marks(self) -> None:
        segments = [
            raw_segment(0.0, 1.0, "Отлично!"),
            raw_segment(1.0, 2.0, "Уверены?"),
        ]

        result = self.aggregator.merge_by_sentences(segments) # type: ignore

        assert len(result) == 2

    def test_trailing_sentence_without_punctuation_is_kept(self) -> None:
        segments = [
            raw_segment(0.0, 1.0, "Законченное предложение."),
            raw_segment(1.0, 2.0, "А это без точки в конце"),
        ]

        result = self.aggregator.merge_by_sentences(segments) # type: ignore

        assert len(result) == 2
        assert result[1].text == "А это без точки в конце"

    def test_empty_input_returns_empty_list(self) -> None:
        assert self.aggregator.merge_by_sentences([]) == []


class TestSegmentAggregatorContractWithFasterWhisper:
    """
    Контрактный тест: остальные тесты в этом файле используют SimpleNamespace
    вместо настоящих Segment/Word — это осознанно, так как merge_by_sentences работает
    через getattr(), а не isinstance().

    Но это же означает, что юнит-тесты не заметят, если faster-whisper
    когда-нибудь переименует поля start/end/text/word. Этот тест про то,
    что реальный класс Word из установленной версии faster-whisper 
    всё ещё имеет ожидаемые атрибуты. Требует установленного faster-whisper.
    """

    def test_real_word_object_has_expected_attributes(self) -> None:
        from faster_whisper.transcribe import Word

        word = Word(start=0.0, end=1.0, word="тест", probability=0.9)
        aggregator = SegmentAggregator()

        # Если это упадёт — значит faster-whisper поменял контракт,
        # и merge_by_sentences нужно чинить, а не только этот тест.
        result = aggregator.merge_by_sentences([word])

        assert result[0].text == "тест"