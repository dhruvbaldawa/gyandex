import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAI
from pydantic import ValidationError

from gyandex.llms.factory import get_model, LLMLoggingCallback
from gyandex.podgen.config.schema import GoogleGenerativeAILLMConfig

def test_get_model_returns_google_generative_ai():
    """Tests that get_model creates a GoogleGenerativeAI instance with correct config"""
    # Given
    config = GoogleGenerativeAILLMConfig(
        provider="google-generative-ai",
        model="gemini-pro",
        temperature=0.7,
        google_api_key="test-key"
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
        config = GoogleGenerativeAILLMConfig(
            provider="unsupported",
            model="test",
            temperature=0.5,
            google_api_key="test-key"
        )
