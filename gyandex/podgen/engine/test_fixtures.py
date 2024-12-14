from unittest.mock import Mock, patch

import pytest

from ..storage.s3 import S3CompatibleStorage
from .publisher import PodcastPublisher


@pytest.fixture
def mock_storage():
    return Mock(spec=S3CompatibleStorage)


@pytest.fixture
def orchestrator(mock_storage, test_db):
    return PodcastPublisher(
        storage=mock_storage,
        db=test_db,
        base_url="https://example.com",
        audio_prefix="episodes",
        feed_prefix="feeds",
    )


@pytest.fixture
def sample_audio(tmp_path):
    audio_path = tmp_path / "test.mp3"
    audio_path.write_bytes(b"fake mp3 content")
    return str(audio_path)


@pytest.fixture
def mock_mutagen():
    with patch("mutagen.File") as mock_file:
        mock_audio = Mock()
        mock_audio.info.length = 300
        mock_audio.mime = ["audio/mpeg"]
        mock_file.return_value = mock_audio
        yield mock_file
