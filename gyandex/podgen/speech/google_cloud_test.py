from unittest.mock import Mock, patch

import pytest
from google.cloud import texttospeech

from ..config.schema import Gender, Participant
from ..speech.google_cloud import GoogleTTSEngine
from ..workflows.types import DialogueLine

dummy_participants = [
    Participant(name="HOST1", language_code="en-US", voice="en-US-Neural2-F", gender=Gender.FEMALE),
    Participant(name="HOST2", language_code="en-US", voice="en-US-Neural2-F", gender=Gender.FEMALE),
]


def test_tts_engine_initialization():
    """Tests that TTSEngine initializes with correct voice configurations"""
    # Given/When
    engine = GoogleTTSEngine(participants=dummy_participants)

    # Then
    assert "HOST1" in engine.voices
    assert "HOST2" in engine.voices
    assert isinstance(engine.client, texttospeech.TextToSpeechClient)


@patch("google.cloud.texttospeech.TextToSpeechClient")
def test_synthesize_speech_for_host1(mock_client):
    """Tests speech synthesis for HOST1 voice"""
    # Given
    engine = GoogleTTSEngine(participants=dummy_participants)
    mock_response = Mock()
    mock_response.audio_content = b"test_audio_content"
    mock_client.return_value.synthesize_speech.return_value = mock_response

    # When
    result = engine.synthesize_speech("Test text", "HOST1")

    # Then
    assert result == b"test_audio_content"
    mock_client.return_value.synthesize_speech.assert_called_once()


@patch("google.cloud.texttospeech.TextToSpeechClient")
def test_process_segment(mock_client):
    """Tests processing of a complete podcast segment"""
    # Given
    engine = GoogleTTSEngine(participants=dummy_participants)
    segment = DialogueLine(text="Test segment", speaker="HOST1")
    mock_response = Mock()
    mock_response.audio_content = b"test_audio_content"
    mock_client.return_value.synthesize_speech.return_value = mock_response

    # When
    result = engine.process_segment(segment)

    # Then
    assert result == b"test_audio_content"


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
    engine = GoogleTTSEngine(participants=dummy_participants)

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
    engine = GoogleTTSEngine(participants=dummy_participants)

    # When
    result = engine.clean_text_for_tts(input_text)

    # Then
    assert result == expected_output, f"Failed test: {description}"
