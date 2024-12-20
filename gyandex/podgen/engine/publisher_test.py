import pytest

from .publisher import PodcastMetadata


def test_create_feed(orchestrator, mock_storage):
    """
    Given: Feed details and a configured orchestrator
    When: Creating a new feed
    Then: Feed should be created in DB and XML uploaded to storage
    """
    # Given
    feed_data = {
        "slug": "test-feed",
        "title": "Test Feed",
        "description": "Test Description",
        "author": "Test Author",
        "email": "test@example.com",
        "website": "https://example.com",
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
        slug="test-feed",
        title="Test Feed",
        description="Test Description",
        author="Test Author",
        email="test@example.com",
        website="https://example.com",
    )
    # Reset the mock after create_feed
    mock_storage.upload_file.reset_mock()

    metadata = PodcastMetadata(
        title="Test Episode",
        description="Test Episode Description",
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
    metadata = PodcastMetadata(title="Test Episode", description="Test Episode Description")

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
        slug="test-feed",
        title="Test Feed",
        description="Test Description",
        author="Test Author",
        email="test@example.com",
        website="https://example.com",
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
