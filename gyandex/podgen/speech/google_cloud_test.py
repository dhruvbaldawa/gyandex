from unittest.mock import Mock, patch
from google.cloud import texttospeech
from gyandex.podgen.processors.tts import GoogleTTSEngine
from gyandex.podgen.engine.workflows import ScriptSegment


def test_tts_engine_initialization():
    """Tests that TTSEngine initializes with correct voice configurations"""
    # Given/When
    engine = GoogleTTSEngine()

    # Then
    assert "HOST1" in engine.voices
    assert "HOST2" in engine.voices
    assert isinstance(engine.client, texttospeech.TextToSpeechClient)


@patch("google.cloud.texttospeech.TextToSpeechClient")
def test_synthesize_speech_for_host1(mock_client):
    """Tests speech synthesis for HOST1 voice"""
    # Given
    engine = GoogleTTSEngine()
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
    engine = GoogleTTSEngine()
    segment = ScriptSegment(dialogue="Test segment", speaker="HOST1")
    mock_response = Mock()
    mock_response.audio_content = b"test_audio_content"
    mock_client.return_value.synthesize_speech.return_value = mock_response

    # When
    result = engine.process_segment(segment)

    # Then
    assert result == b"test_audio_content"
