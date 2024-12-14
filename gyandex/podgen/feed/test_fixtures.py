import os

import pytest

from .models import PodcastDB


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
        "slug": "test-podcast",
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
        "episode_type": "full",
    }
