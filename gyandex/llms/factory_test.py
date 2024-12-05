import pytest
from langchain_google_genai import GoogleGenerativeAI
from pydantic import ValidationError

from ..podgen.config.schema import GoogleGenerativeAILLMConfig
from .factory import get_model


def test_get_model_returns_google_generative_ai():
    """Tests that get_model creates a GoogleGenerativeAI instance with correct config"""
    # Given
    config = GoogleGenerativeAILLMConfig(
        provider="google-generative-ai", model="gemini-pro", temperature=0.7, google_api_key="test-key"
    )

    # When
    model = get_model(config, "/tmp")

    # Then
    assert isinstance(model, GoogleGenerativeAI)
    assert model.model == "gemini-pro"
    assert model.temperature == 0.7


def test_get_model_raises_for_unsupported_provider():
    """Tests that get_model raises NotImplementedError for unsupported providers"""
    # When/Then
    with pytest.raises(ValidationError):
        _ = GoogleGenerativeAILLMConfig(
            provider="unsupported", model="test", temperature=0.5, google_api_key="test-key"
        )
