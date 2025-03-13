from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from pydub import AudioSegment

from ..config.schema import Gender, Participant
from ..workflows.types import DialogueLine
from .zyphra import ZyphraTTSEngine


@pytest.fixture
def participants():
    return [
        Participant(name="HOST1", language_code="en-US", voice="en-us-female-1", gender=Gender.FEMALE),
        Participant(name="HOST2", language_code="en-US", voice="en-us-male-1", gender=Gender.MALE),
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
        language_code="en-US",
        output_format="mp3"
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
        
        mock_from_mp3.side_effect = [mock_segment1, mock_segment2]
        mock_empty.return_value = mock_combined
        
        # When
        engine.generate_audio_file(audio_segments, podcast_path)
        
        # Then
        assert mock_from_mp3.call_count == 2
        mock_from_mp3.assert_any_call(BytesIO(b"segment1"))
        mock_from_mp3.assert_any_call(BytesIO(b"segment2"))
        
        # Check that export was called
        mock_combined.export.assert_called_once_with(podcast_path, format="mp3")
