from unittest.mock import Mock, patch

import pytest

from .s3 import S3CompatibleStorage


@pytest.fixture
def mock_s3_factory():
    with patch("boto3.client") as mock_client:
        # Create a mock client instance
        client = Mock()
        mock_client.return_value = client

        # Mock the meta attributes
        client.meta.endpoint_url = None
        client.meta.region_name = "us-east-1"

        yield mock_client, client


@pytest.fixture
def mock_s3_storage(mock_s3_factory):
    mock_client, _ = mock_s3_factory
    return mock_client


@pytest.fixture
def mock_s3_client(mock_s3_factory):
    _, client = mock_s3_factory
    return client


@pytest.fixture
def storage(mock_s3_client):
    return S3CompatibleStorage(
        bucket="test-bucket",
        access_key_id="test-key",
        secret_access_key="test-secret",
        region_name="us-east-1",
    )


@pytest.fixture
def r2_storage(mock_s3_client):
    # Mock R2 endpoint
    mock_s3_client.meta.endpoint_url = "https://test.r2.cloudflarestorage.com"
    return S3CompatibleStorage(
        bucket="test-bucket",
        access_key_id="test-key",
        secret_access_key="test-secret",
        endpoint_url="https://test.r2.cloudflarestorage.com",
        region_name="auto",
    )
