import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydub import AudioSegment

from ..config.schema import Gender, Participant
from ..workflows.types import DialogueLine
from .zyphra import ZyphraTTSEngine


@pytest.fixture
def participants():
    return [
        Participant(name="HOST1", language_code="en-us", voice="en-us-female-1", gender=Gender.FEMALE),
        Participant(name="HOST2", language_code="en-us", voice="en-us-male-1", gender=Gender.MALE),
    ]


@pytest.fixture
def mock_zyphra_client():
    with patch("zyphra.ZyphraClient") as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        # Setup the nested structure for audio.speech.create
        audio = MagicMock()
        speech = MagicMock()
        create = MagicMock(return_value=b"mock_audio_data")

        speech.create = create
        audio.speech = speech
        client_instance.audio = audio

        yield mock_client


def test_zyphra_tts_engine_initialization(participants, mock_zyphra_client):
    """Tests that ZyphraTTSEngine initializes correctly"""
    # Given/When
    engine = ZyphraTTSEngine(participants, api_key="test-api-key")

    # Then
    mock_zyphra_client.assert_called_once_with(api_key="test-api-key")
    assert "HOST1" in engine.voices
    assert "HOST2" in engine.voices
    assert engine.voices["HOST1"]["voice"] == "en-us-female-1"
    assert engine.voices["HOST1"]["gender"] == "female"
    assert engine.voices["HOST2"]["voice"] == "en-us-male-1"
    assert engine.voices["HOST2"]["gender"] == "male"


def test_zyphra_tts_engine_process_segment(participants, mock_zyphra_client):
    """Tests that process_segment calls synthesize_speech with correct parameters"""
    # Given
    engine = ZyphraTTSEngine(participants, api_key="test-api-key")
    segment = DialogueLine(speaker="HOST1", text="Hello, world!")

    # When
    with patch.object(engine, "synthesize_speech", return_value=b"mock_audio_data") as mock_synthesize:
        result = engine.process_segment(segment)

    # Then
    mock_synthesize.assert_called_once_with("Hello, world!", "HOST1")
    assert result == b"mock_audio_data"


def test_zyphra_tts_engine_synthesize_speech(participants, mock_zyphra_client):
    """Tests that synthesize_speech calls the Zyphra API with correct parameters"""
    # Given
    engine = ZyphraTTSEngine(participants, api_key="test-api-key")

    # When
    result = engine.synthesize_speech("Hello, world!", "HOST1")

    # Then
    client = mock_zyphra_client.return_value
    client.audio.speech.create.assert_called_once_with(
        text="Hello, world!",
        voice="en-us-female-1",
        language_code="en-us",
        mime_type="audio/mp3"
    )
    assert result == b"mock_audio_data"


def test_zyphra_tts_engine_generate_audio_file(participants, mock_zyphra_client):
    """Tests that generate_audio_file combines audio segments correctly"""
    # Given
    engine = ZyphraTTSEngine(participants, api_key="test-api-key")
    audio_segments = [b"segment1", b"segment2"]
    podcast_path = "test_podcast.mp3"

    # Mock AudioSegment functionality
    with patch("pydub.AudioSegment.from_mp3") as mock_from_mp3, \
         patch("pydub.AudioSegment.empty") as mock_empty:

        # Setup mocks
        mock_segment1 = MagicMock()
        mock_segment2 = MagicMock()
        mock_combined = MagicMock()

        # Configure the mocks to handle the += and append operations
        mock_combined.__iadd__ = MagicMock(return_value=mock_combined)
        mock_combined.append = MagicMock(return_value=mock_combined)

        mock_from_mp3.side_effect = [mock_segment1, mock_segment2]
        mock_empty.return_value = mock_combined

        # When
        engine.generate_audio_file(audio_segments, podcast_path)

        # Then
        assert mock_from_mp3.call_count == 2

        # Instead of comparing BytesIO objects directly, check the content
        call_args_list = mock_from_mp3.call_args_list
        assert len(call_args_list) == 2

        # Extract the BytesIO objects passed to from_mp3
        bytesio_args = [args[0] for args, _ in call_args_list]

        # Read their content and verify
        contents = [bio.getvalue() for bio in bytesio_args]
        assert b"segment1" in contents
        assert b"segment2" in contents

        # Check that export was called
        mock_combined.export.assert_called_once_with(podcast_path, format="mp3")


def test_zyphra_tts_engine_synthesize_speech_with_path_return(participants):
    """Tests that synthesize_speech correctly handles Path object returns from Zyphra API"""
    # Create a temp file with test audio content
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b"mock_audio_from_path")
        temp_path = Path(temp_file.name)

    # Setup the mock to return a Path object
    with patch("zyphra.ZyphraClient") as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        # Setup the nested structure for audio.speech.create to return a Path
        audio = MagicMock()
        speech = MagicMock()
        create = MagicMock(return_value=temp_path)

        speech.create = create
        audio.speech = speech
        client_instance.audio = audio

        # When
        engine = ZyphraTTSEngine(participants, api_key="test-api-key")
        result = engine.synthesize_speech("Hello, world!", "HOST1")

        # Then
        client_instance.audio.speech.create.assert_called_once_with(
            text="Hello, world!",
            voice="en-us-female-1",
            language_code="en-us",
            mime_type="audio/mp3"
        )
        assert isinstance(result, bytes)
        assert result == b"mock_audio_from_path"

    # Clean up the temp file
    temp_path.unlink(missing_ok=True)


def test_zyphra_tts_engine_synthesize_speech_with_string_path_return(participants):
    """Tests that synthesize_speech correctly handles string path returns from Zyphra API"""
    # Create a temp file with test audio content
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b"mock_audio_from_string_path")
        temp_path_str = temp_file.name

    # Setup the mock to return a string path
    with patch("zyphra.ZyphraClient") as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        # Setup the nested structure for audio.speech.create to return a string path
        audio = MagicMock()
        speech = MagicMock()
        create = MagicMock(return_value=temp_path_str)

        speech.create = create
        audio.speech = speech
        client_instance.audio = audio

        # When
        engine = ZyphraTTSEngine(participants, api_key="test-api-key")
        result = engine.synthesize_speech("Hello, world!", "HOST1")

        # Then
        client_instance.audio.speech.create.assert_called_once_with(
            text="Hello, world!",
            voice="en-us-female-1",
            language_code="en-us",
            mime_type="audio/mp3"
        )
        assert isinstance(result, bytes)
        assert result == b"mock_audio_from_string_path"

    # Clean up the temp file
    from pathlib import Path
    Path(temp_path_str).unlink(missing_ok=True)


def test_zyphra_tts_engine_synthesize_speech_with_fallback(participants):
    """Tests that synthesize_speech uses the fallback conversion for unexpected return types"""
    # Given
    class CustomBytesLike:
        def __bytes__(self):
            return b"converted_to_bytes"

    # Setup the mock to return a custom object that can be converted to bytes
    with patch("zyphra.ZyphraClient") as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        # Setup the nested structure for audio.speech.create to return a custom object
        audio = MagicMock()
        speech = MagicMock()
        create = MagicMock(return_value=CustomBytesLike())

        speech.create = create
        audio.speech = speech
        client_instance.audio = audio

        # When
        engine = ZyphraTTSEngine(participants, api_key="test-api-key")
        result = engine.synthesize_speech("Hello, world!", "HOST1")

        # Then
        client_instance.audio.speech.create.assert_called_once_with(
            text="Hello, world!",
            voice="en-us-female-1",
            language_code="en-us",
            mime_type="audio/mp3"
        )
        assert isinstance(result, bytes)
        assert result == b"converted_to_bytes"
