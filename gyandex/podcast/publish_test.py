import pytest
from datetime import datetime
from unittest.mock import Mock, patch
import os
from .publish import PodcastOrchestrator, PodcastMetadata
from .models_test import test_db  # @todo: move to common fixtures
from .storage import S3CompatibleStorage


@pytest.fixture
def mock_storage():
    return Mock(spec=S3CompatibleStorage)


@pytest.fixture
def orchestrator(mock_storage, test_db):
    return PodcastOrchestrator(
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


def test_create_feed(orchestrator, mock_storage):
    """
    Given: Feed details and a configured orchestrator
    When: Creating a new feed
    Then: Feed should be created in DB and XML uploaded to storage
    """
    # Given
    feed_data = {
        "name": "test-feed",
        "title": "Test Feed",
        "description": "Test Description",
        "author": "Test Author",
        "email": "test@example.com",
    }
    mock_storage.upload_file.return_value = "https://example.com/feeds/test-feed.xml"

    # When
    feed_url = orchestrator.create_feed(**feed_data)

    # Then
    assert feed_url == "https://example.com/feeds/test-feed.xml"
    mock_storage.upload_file.assert_called_once()
    assert orchestrator.db.get_feed("test-feed") is not None


def test_add_episode(orchestrator, mock_storage, sample_audio, mock_mutagen):
    """
    Given: An existing feed and episode metadata
    When: Adding a new episode
    Then: Episode should be added to DB and files uploaded to storage
    """
    # Given
    orchestrator.create_feed(
        name="test-feed",
        title="Test Feed",
        description="Test Description",
        author="Test Author",
        email="test@example.com",
    )
    # Reset the mock after create_feed
    mock_storage.upload_file.reset_mock()

    metadata = PodcastMetadata(
        title="Test Episode", description="Test Episode Description", episode_number=1
    )

    mock_storage.upload_file.side_effect = [
        "https://example.com/episodes/test.mp3",
        "https://example.com/feeds/test-feed.xml",
    ]

    # When
    result = orchestrator.add_episode("test-feed", sample_audio, metadata)

    # Then
    assert result["episode_url"] == "https://example.com/episodes/test.mp3"
    assert result["feed_url"] == "https://example.com/feeds/test-feed.xml"
    assert len(mock_storage.upload_file.call_args_list) == 2


def test_add_episode_to_nonexistent_feed(orchestrator, sample_audio):
    """
    Given: A non-existent feed
    When: Attempting to add an episode
    Then: ValueError should be raised
    """
    # Given
    metadata = PodcastMetadata(
        title="Test Episode", description="Test Episode Description"
    )

    # When/Then
    with pytest.raises(ValueError):
        orchestrator.add_episode("nonexistent-feed", sample_audio, metadata)


def test_list_episodes(orchestrator, mock_storage, sample_audio, mock_mutagen):
    """
    Given: A feed with multiple episodes
    When: Listing episodes
    Then: Episodes should be returned in correct order
    """
    # Given
    orchestrator.create_feed(
        name="test-feed",
        title="Test Feed",
        description="Test Description",
        author="Test Author",
        email="test@example.com",
    )

    mock_storage.upload_file.return_value = "https://example.com/episodes/test.mp3"

    # Create two episodes with different content to generate unique GUIDs
    with open(sample_audio, "wb") as f:
        f.write(b"episode1 content")
    metadata1 = PodcastMetadata(title="Episode 1", description="First episode")
    orchestrator.add_episode("test-feed", sample_audio, metadata1)

    with open(sample_audio, "wb") as f:
        f.write(b"episode2 content")
    metadata2 = PodcastMetadata(title="Episode 2", description="Second episode")
    orchestrator.add_episode("test-feed", sample_audio, metadata2)

    # When
    episodes = orchestrator.list_episodes("test-feed")

    # Then
    assert len(episodes) == 2
    assert episodes[0].title == "Episode 2"  # Most recent first
    assert episodes[1].title == "Episode 1"


def test_get_feed_url(orchestrator):
    """
    Given: A feed name
    When: Getting the feed URL
    Then: Correct URL should be returned
    """
    # When
    feed_url = orchestrator.get_feed_url("test-feed")

    # Then
    assert feed_url == "https://example.com/feeds/test-feed.xml"
