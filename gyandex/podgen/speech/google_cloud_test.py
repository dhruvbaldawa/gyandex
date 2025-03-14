from unittest.mock import Mock, patch

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
