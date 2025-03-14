from unittest.mock import MagicMock, patch

from ..config.schema import Gender, Participant
from ..workflows.types import DialogueLine
from .openai import OpenAITTSEngine


@patch("openai.OpenAI")
def test_openai_tts_engine_initialization(mock_openai):
    """Test proper initialization of the OpenAI TTS engine"""
    # Given
    participants = [
        Participant(name="Sarah", voice="nova", gender=Gender.FEMALE),
        Participant(name="Mike", voice="onyx", gender=Gender.MALE),
    ]
    mock_openai.return_value = MagicMock()  # Mock the OpenAI client
    # When
    engine = OpenAITTSEngine(participants, model="tts-1", api_key="test-key")
    # Then
    assert engine.model == "tts-1"
    assert "Sarah" in engine.voices
    assert "Mike" in engine.voices
    assert engine.voices["Sarah"] == "nova"
    assert engine.voices["Mike"] == "onyx"


@patch("openai.OpenAI")
def test_openai_tts_engine_voice_defaults(mock_openai):
    """Test that default voices are mapped correctly when custom voices are not specified"""
    # Given
    participants = [
        Participant(name="Host1", gender=Gender.FEMALE, voice=""),
        Participant(name="Host2", gender=Gender.MALE, voice=""),
        Participant(name="Host3", gender=Gender.NON_BINARY, voice=""),
    ]
    mock_openai.return_value = MagicMock()  # Mock the OpenAI client
    # When
    engine = OpenAITTSEngine(participants, api_key="test-key")
    # Then
    assert engine.voices["Host1"] == "nova"  # Default female voice
    assert engine.voices["Host2"] == "onyx"  # Default male voice
    assert engine.voices["Host3"] == "alloy"  # Default non-binary voice


@patch("openai.OpenAI")
def test_clean_text_for_tts(mock_openai):
    """Test that text cleaning works correctly for TTS preparation"""
    # Given
    engine = OpenAITTSEngine([Participant(name="Host", gender=Gender.MALE, voice="")], api_key="test-key")
    text = (
        "This is a *test* with _emphasis_ and **bold** text. This is a 5*3 calculation. Wait, *hold on*—let me rethink."
    )
    mock_openai.return_value = MagicMock()  # Mock the OpenAI client
    # When
    cleaned = engine.clean_text_for_tts(text)
    # Then
    assert "*" not in cleaned
    assert "_" not in cleaned
    assert "5 times 3" in cleaned
    assert "Wait, hold on" in cleaned
    assert "—" not in cleaned


@patch("gyandex.podgen.speech.openai.OpenAI")
def test_process_segment(mock_openai_class):
    """Test that process_segment properly calls the OpenAI API with cleaned text"""
    # Given
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.content = b"audio data"
    mock_client.audio.speech.create.return_value = mock_response

    participants = [Participant(name="Host", voice="alloy", gender=Gender.MALE)]
    engine = OpenAITTSEngine(participants, api_key="test-key")

    segment = DialogueLine(speaker="Host", text="This is a *test*.")

    # When
    result = engine.process_segment(segment)

    # Then
    assert result == b"audio data"
    # Verify the call was made with cleaned text
    mock_client.audio.speech.create.assert_called_once()
    call_kwargs = mock_client.audio.speech.create.call_args.kwargs
    assert call_kwargs["input"] == "This is a test."
    assert call_kwargs["voice"] == "alloy"
    assert call_kwargs["model"] == "tts-1"


@patch("gyandex.podgen.speech.openai.OpenAI")
def test_synthesize_speech(mock_openai):
    """Test that synthesize_speech properly calls the OpenAI API"""
    # Given
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    mock_response = MagicMock()
    mock_response.content = b"audio data"
    mock_client.audio.speech.create.return_value = mock_response

    participants = [Participant(name="Host", voice="alloy", gender=Gender.MALE)]
    engine = OpenAITTSEngine(participants, api_key="test-key")

    # When
    result = engine.synthesize_speech("Test text", "Host")

    # Then
    assert result == b"audio data"
    mock_client.audio.speech.create.assert_called_once_with(model="tts-1", voice="alloy", input="Test text")
