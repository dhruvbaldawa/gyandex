import pytest
from pydantic import ValidationError

from gyandex.podgen.config.schema import S3StorageConfig
from gyandex.podgen.storage.factory import get_storage
from gyandex.podgen.storage.s3 import S3CompatibleStorage

def test_get_storage_returns_s3_storage():
    """Tests that get_storage creates an S3CompatibleStorage instance with correct config"""
    # Given
    config = S3StorageConfig(
        provider="s3",
        bucket="test-bucket",
        access_key="test-access-key",
        secret_key="test-secret-key",
        region="us-east-1",
        endpoint="https://test-endpoint",
        custom_domain="cdn.example.com"
    )

    # When
    storage = get_storage(config)

    # Then
    assert isinstance(storage, S3CompatibleStorage)
    assert storage.bucket == "test-bucket"
    assert storage.custom_domain == "cdn.example.com"

def test_get_storage_raises_for_unsupported_provider():
    """Tests that get_storage raises NotImplementedError for unsupported providers"""
    # When/Then
    with pytest.raises(ValidationError):
        config = S3StorageConfig(
            provider="unsupported",
            bucket="test-bucket",
            access_key="test-access-key",
            secret_key="test-secret-key",
            region="us-east-1",
            endpoint="https://test-endpoint",
            custom_domain="cdn.example.com"
        )
