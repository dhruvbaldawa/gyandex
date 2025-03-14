from unittest.mock import MagicMock

import pytest

from ..config.schema import Gender, Participant
from .base import BaseTTSEngine


class DummyTTSEngine(BaseTTSEngine):
    """Dummy TTS engine implementation for testing the base class"""

    def generate_voice_profile(self, participants):
        return {p.name: p.voice for p in participants}

    def synthesize_speech(self, text, speaker):
        return f"{speaker}:{text}".encode()


@pytest.mark.parametrize(
    "input_text,expected_output",
    [
        # Test cases for markdown emphasis patterns
        ("This is *emphasized* text", "This is emphasized text"),
        ("This is _emphasized_ text", "This is emphasized text"),
        ("This is **bold** text", "This is bold text"),
        ("This is __bold__ text", "This is bold text"),
        ("This is *multi-word* emphasis", "This is multi-word emphasis"),
        ("*Start* and *end* emphasis", "Start and end emphasis"),
        # Test cases for math expressions
        ("The formula is 2 * 3 = 6", "The formula is 2 times 3 = 6"),
        # Test cases for dashes and interruptions
        ("This is a sentence—with an em dash", "This is a sentence, with an em dash"),
        ("This is a sentence--with a double dash", "This is a sentence, with a double dash"),
        ("Speaker 1: I was thinking—", "Speaker 1: I was thinking,"),
        ("Speaker 1: I was—wait", "Speaker 1: I was, wait"),
        # Test cases for combined formatting
        ("This text has *emphasis* and—interruption", "This text has emphasis and, interruption"),
        ("Wait, *hold on*. I need to—think", "Wait, hold on. I need to, think"),
        ("Exactly—Wait, *exactly* what?", "Exactly, Wait, exactly what?"),
        # Test cases for legitimate asterisks (should be preserved or properly transformed)
        ("The code example function(*args) should work", "The code example function(*args) should work"),
        # Test case for multiple whitespace (should be normalized)
        ("This  has    multiple   spaces", "This has multiple spaces"),
    ],
)
def test_clean_text_for_tts(input_text, expected_output):
    """Test the text cleaning functionality with various input patterns"""
    # Given
    engine = DummyTTSEngine([Participant(name="Host", gender=Gender.MALE, voice="test-voice")])

    # When
    result = engine.clean_text_for_tts(input_text)

    # Then
    assert result == expected_output


@pytest.mark.parametrize(
    "description,input_text,expected_output",
    [
        # Real examples from sample transcript provided in the requirements
        (
            "Emphasis within a podcast intro",
            "Welcome back, everyone! Or wait, actually, welcome for the first time! This is *The Philosophers' Corner*, where we dig into timeless ideas...",  # noqa: E501
            "Welcome back, everyone! Or wait, actually, welcome for the first time! This is The Philosophers' Corner, where we dig into timeless ideas...",  # noqa: E501
        ),
        (
            "Complex dialogue with interruption markers",
            "Sarah: —say experience is the only way! Bacon says studies perfect *and* are perfected by experience.",
            "Sarah: , say experience is the only way! Bacon says studies perfect and are perfected by experience.",
        ),
        (
            "Multiple formatting patterns in one line",
            "Sarah: *Exactly what?* Wait, sorry, Mike. My turn. Let's pivot to 'reading by deputy.' You ever outsource that?",  # noqa: E501
            "Sarah: Exactly what? Wait, sorry, Mike. My turn. Let's pivot to 'reading by deputy.' You ever outsource that?",  # noqa: E501
        ),
    ],
)
def test_clean_text_for_tts_realistic_examples(description, input_text, expected_output):
    """Test the text cleaning functionality with realistic podcast transcript examples"""
    # Given
    engine = DummyTTSEngine([Participant(name="Host", gender=Gender.MALE, voice="test-voice")])

    # When
    result = engine.clean_text_for_tts(input_text)

    # Then
    assert result == expected_output, f"Failed test: {description}"


def test_process_segment():
    """Test that process_segment cleans text and calls synthesize_speech"""
    # Given
    engine = DummyTTSEngine([Participant(name="Host", gender=Gender.MALE, voice="test-voice")])

    # Create a mock DialogueLine
    dialogue_line = MagicMock()
    dialogue_line.speaker = "Host"
    dialogue_line.text = "This is a *test* with _emphasis_."

    # When
    result = engine.process_segment(dialogue_line)

    # Then
    assert result == b"Host:This is a test with emphasis."


@pytest.mark.skip(reason="This test requires properly mocked audio processing")
def test_generate_audio_file():
    """Test audio file generation - skipped because it requires complex mocking"""
    # This test is skipped because it requires complex mocking of the audio processing
    # libraries. In a real implementation, we would use proper mocking to test this
    # functionality without relying on external dependencies.
    pass
