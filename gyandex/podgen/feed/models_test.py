import os
import pytest

from .models import PodcastDB, Feed


@pytest.fixture
def test_db():
    """Create a temporary test database"""
    db_path = "test_podcast.db"
    db = PodcastDB(db_path)
    yield db
    os.remove(db_path)


@pytest.fixture
def db_session(test_db):
    """Create a database session for testing"""
    Session = test_db.session
    with Session() as session:
        yield session


@pytest.fixture
def sample_feed_data():
    """Sample feed data for testing"""
    return {
        "name": "test-podcast",
        "title": "Test Podcast",
        "description": "A test podcast",
        "author": "Test Author",
        "email": "test@example.com",
        "website": "https://example.com",
        "language": "en",
        "copyright": "2024 Test Author",
        "categories": "Technology,Education",
        "explicit": "no",
    }


@pytest.fixture
def sample_episode_data():
    """Sample episode data for testing"""
    return {
        "title": "Test Episode",
        "description": "A test episode",
        "audio_url": "https://example.com/episode1.mp3",
        "guid": "episode-1",
        "duration": 1800,
        "file_size": 15000000,
        "mime_type": "audio/mpeg",
        "episode_number": 1,
        "season_number": 1,
        "episode_type": "full",
    }


# Database Tests
def test_create_feed(test_db, db_session, sample_feed_data):
    """
    Given: A new podcast feed data
    When: Creating a feed in the database
    Then: The feed should be stored with correct attributes
    """
    # When
    feed = test_db.create_feed(**sample_feed_data)

    # Then
    stored_feed = db_session.query(Feed).filter(Feed.id == feed.id).first()

    assert stored_feed.name == sample_feed_data["name"]
    assert stored_feed.title == sample_feed_data["title"]
    assert stored_feed.author == sample_feed_data["author"]
    assert stored_feed.created_at is not None


def test_get_nonexistent_feed(test_db):
    """
    Given: An empty database
    When: Attempting to get a non-existent feed
    Then: None should be returned
    """
    # When
    feed = test_db.get_feed("nonexistent")

    # Then
    assert feed is None


def test_add_episode_to_feed(test_db, sample_feed_data, sample_episode_data):
    """
    Given: An existing feed in the database
    When: Adding a new episode to the feed
    Then: The episode should be stored with correct attributes
    """
    # Given
    feed = test_db.create_feed(**sample_feed_data)

    # When
    episode = test_db.add_episode(feed_name=feed.name, **sample_episode_data)

    # Then
    assert episode.title == sample_episode_data["title"]
    assert episode.guid == sample_episode_data["guid"]
    assert episode.duration == sample_episode_data["duration"]
    assert episode.feed_id == feed.id


def test_add_episode_to_nonexistent_feed(test_db, sample_episode_data):
    """
    Given: A database with no feeds
    When: Attempting to add an episode to a non-existent feed
    Then: A ValueError should be raised
    """
    # When/Then
    with pytest.raises(ValueError):
        test_db.add_episode(feed_name="nonexistent", **sample_episode_data)


def test_get_episodes_ordered_by_date(test_db, sample_feed_data, sample_episode_data):
    """
    Given: A feed with multiple episodes
    When: Retrieving episodes
    Then: Episodes should be ordered by publication date descending
    """
    # Given
    feed = test_db.create_feed(**sample_feed_data)

    # Create episodes with different dates
    episode1_data = {**sample_episode_data, "guid": "ep1"}
    episode2_data = {**sample_episode_data, "guid": "ep2"}

    test_db.add_episode(feed.name, **episode1_data)
    test_db.add_episode(feed.name, **episode2_data)

    # When
    episodes = test_db.get_episodes(feed.name)

    # Then
    assert len(episodes) == 2
    assert episodes[0].publication_date >= episodes[1].publication_date
