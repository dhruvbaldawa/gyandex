from unittest.mock import ANY, Mock

import pytest
from botocore.exceptions import ClientError

from .s3 import S3CompatibleStorage


def test_initialization(mock_s3_storage):
    """Test storage initialization with different configurations"""
    # Test AWS S3 initialization
    _ = S3CompatibleStorage(bucket="test-bucket", access_key_id="test-key", secret_access_key="test-secret")

    mock_s3_storage.assert_called_once_with(
        "s3",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        endpoint_url=None,
        region_name="auto",
        config=ANY,
    )

    mock_s3_storage.reset_mock()

    # Test R2 initialization
    _ = S3CompatibleStorage(
        bucket="test-bucket",
        access_key_id="test-key",
        secret_access_key="test-secret",
        endpoint_url="https://test.r2.cloudflarestorage.com",
    )

    mock_s3_storage.assert_called_once_with(
        "s3",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        endpoint_url="https://test.r2.cloudflarestorage.com",
        region_name="auto",
        config=ANY,
    )


def test_upload_file(storage, mock_s3_client, tmp_path):
    """Test file upload functionality"""
    # Create a temporary file
    test_file = tmp_path / "test.mp3"
    test_file.write_bytes(b"test content")

    storage.upload_file(str(test_file), "episodes/test.mp3", metadata={"episode": "1"})

    mock_s3_client.upload_file.assert_called_with(
        str(test_file),
        "test-bucket",
        "episodes/test.mp3",
        ExtraArgs={
            "ACL": "public-read",
            "ContentType": "audio/mpeg",
            "Metadata": {"episode": "1"},
        },
    )


def test_download_file(storage, mock_s3_client, tmp_path):
    """Test file download functionality"""
    download_path = tmp_path / "downloaded.mp3"

    storage.download_file("episodes/test.mp3", str(download_path))

    mock_s3_client.download_file.assert_called_with("test-bucket", "episodes/test.mp3", str(download_path))


def test_get_public_url_aws(storage):
    """Test public URL generation for AWS S3"""
    url = storage.get_public_url("episodes/test.mp3")
    expected_url = "https://test-bucket.s3.us-east-1.amazonaws.com/episodes/test.mp3"
    assert url == expected_url


def test_get_public_url_r2(r2_storage):
    """Test public URL generation for R2"""
    url = r2_storage.get_public_url("episodes/test.mp3")
    expected_url = "https://test.r2.cloudflarestorage.com/test-bucket/episodes/test.mp3"
    assert url == expected_url


def test_get_public_url_custom_domain(mock_s3_client):
    """Test public URL generation with custom domain"""
    storage = S3CompatibleStorage(
        bucket="test-bucket",
        access_key_id="test-key",
        secret_access_key="test-secret",
        custom_domain="cdn.example.com",
    )

    url = storage.get_public_url("episodes/test.mp3")
    expected_url = "https://cdn.example.com/episodes/test.mp3"
    assert url == expected_url


def test_list_files(storage, mock_s3_client):
    """Test file listing functionality"""
    # Mock paginator
    paginator = Mock()
    mock_s3_client.get_paginator.return_value = paginator

    # Mock paginator response
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": "episodes/ep1.mp3", "Size": 1000},
                {"Key": "episodes/ep2.mp3", "Size": 2000},
            ]
        }
    ]

    files = storage.list_files(prefix="episodes/")

    assert len(files) == 2
    assert files[0]["Key"] == "episodes/ep1.mp3"
    assert files[1]["Key"] == "episodes/ep2.mp3"

    mock_s3_client.get_paginator.assert_called_with("list_objects_v2")
    paginator.paginate.assert_called_with(Bucket="test-bucket", Prefix="episodes/")


def test_delete_file(storage, mock_s3_client):
    """Test file deletion functionality"""
    storage.delete_file("episodes/test.mp3")

    mock_s3_client.delete_object.assert_called_with(Bucket="test-bucket", Key="episodes/test.mp3")


def test_upload_file_content_type_guessing(storage, mock_s3_client, tmp_path):
    """Test content type guessing for different file types"""
    test_cases = [
        ("test.mp3", "audio/mpeg"),
        ("test.wav", "audio/x-wav"),
        ("test.txt", "text/plain"),
        ("test.unknown", "application/octet-stream"),
    ]

    for filename, expected_content_type in test_cases:
        test_file = tmp_path / filename
        test_file.write_bytes(b"test content")

        storage.upload_file(str(test_file), f"test/{filename}")

        mock_s3_client.upload_file.assert_called_with(
            str(test_file),
            "test-bucket",
            f"test/{filename}",
            ExtraArgs={"ACL": "public-read", "ContentType": expected_content_type},
        )


def test_error_handling(storage, mock_s3_client):
    """Test error handling for various operations"""
    # Mock ClientError for upload
    mock_s3_client.upload_file.side_effect = ClientError(
        {"Error": {"Code": "NoSuchBucket", "Message": "The bucket does not exist"}},
        "upload_file",
    )

    with pytest.raises(ClientError) as exc_info:
        storage.upload_file("test.mp3", "test/test.mp3")
    assert exc_info.value.response["Error"]["Code"] == "NoSuchBucket"

    # Mock ClientError for download
    mock_s3_client.download_file.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist"}},
        "download_file",
    )

    with pytest.raises(ClientError) as exc_info:
        storage.download_file("nonexistent.mp3", "./local.mp3")
    assert exc_info.value.response["Error"]["Code"] == "NoSuchKey"


def test_upload_with_custom_acl(mock_s3_client, tmp_path):
    """Test upload with custom ACL"""
    storage = S3CompatibleStorage(
        bucket="test-bucket",
        access_key_id="test-key",
        secret_access_key="test-secret",
        acl="private",
    )

    test_file = tmp_path / "test.mp3"
    test_file.write_bytes(b"test content")

    storage.upload_file(str(test_file), "test/test.mp3")

    mock_s3_client.upload_file.assert_called_with(
        str(test_file),
        "test-bucket",
        "test/test.mp3",
        ExtraArgs={"ACL": "private", "ContentType": "audio/mpeg"},
    )
