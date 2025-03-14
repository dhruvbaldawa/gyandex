import pytest

from ..config.schema import Gender, GoogleCloudTTSConfig, OpenAITTSConfig, Participant
from .factory import get_text_to_speech_engine
from .google_cloud import GoogleTTSEngine
from .openai import OpenAITTSEngine


def test_get_text_to_speech_engine_returns_google_cloud():
    """Tests that get_text_to_speech_engine creates a GoogleTTSEngine instance with correct config"""
    # Given
    participants = [Participant(name="HOST1", language_code="en-US", voice="en-US-Neural2-F", gender=Gender.FEMALE)]
    config = GoogleCloudTTSConfig(provider="google-cloud", participants=participants)

    # When
    engine = get_text_to_speech_engine(config)

    # Then
    assert isinstance(engine, GoogleTTSEngine)
    assert engine.voices["HOST1"].name == "en-US-Neural2-F"


def test_get_text_to_speech_engine_returns_openai():
    """Tests that get_text_to_speech_engine creates an OpenAITTSEngine instance with correct config"""
    # Given
    participants = [Participant(name="HOST1", language_code="en-US", voice="alloy", gender=Gender.FEMALE)]
    config = OpenAITTSConfig(provider="openai", participants=participants, api_key="test-key")

    # When
    engine = get_text_to_speech_engine(config)

    # Then
    assert isinstance(engine, OpenAITTSEngine)
    assert engine.voices["HOST1"] == "alloy"
    assert engine.model == "tts-1"


def test_get_text_to_speech_engine_raises_for_unsupported_provider():
    """Tests that get_text_to_speech_engine raises NotImplementedError for unsupported providers"""
    # Given
    config = GoogleCloudTTSConfig.model_construct(provider="unsupported", participants=[])

    # When/Then
    with pytest.raises(NotImplementedError, match="Unsupported TTS provider: unsupported"):
        get_text_to_speech_engine(config)
