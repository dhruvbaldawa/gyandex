import xml.etree.ElementTree as ET

import pytest

from .generator import PodcastFeedGenerator

# Feed Generator Tests


def test_generate_feed_xml(test_db, sample_feed_data, sample_episode_data):
    """
    Given: A feed with an episode in the database
    When: Generating the RSS feed
    Then: Valid podcast RSS XML should be generated with correct data
    """
    # Given
    feed = test_db.create_feed(**sample_feed_data)
    _ = test_db.add_episode(feed.slug, **sample_episode_data)

    # When
    generator = PodcastFeedGenerator(test_db)
    feed_xml = generator.generate_feed(feed.slug)

    # Then
    root = ET.fromstring(feed_xml)
    channel = root.find("channel")

    assert channel.find("title").text == sample_feed_data["title"]
    assert channel.find("description").text == sample_feed_data["description"]

    # Check iTunes specific tags
    itunes_ns = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}
    assert channel.find("itunes:author", itunes_ns).text == sample_feed_data["author"]


def test_generate_feed_with_nonexistent_feed(test_db):
    """
    Given: An empty database
    When: Attempting to generate a feed for a non-existent feed
    Then: A ValueError should be raised
    """
    # Given
    generator = PodcastFeedGenerator(test_db)

    # When/Then
    with pytest.raises(ValueError):
        generator.generate_feed("nonexistent")


def test_feed_episode_enclosure(test_db, sample_feed_data, sample_episode_data):
    """
    Given: A feed with an episode
    When: Generating the RSS feed
    Then: Episode enclosure should have correct attributes
    """
    # Given
    feed = test_db.create_feed(**sample_feed_data)
    _ = test_db.add_episode(feed.slug, **sample_episode_data)

    # When
    generator = PodcastFeedGenerator(test_db)
    feed_xml = generator.generate_feed(feed.slug)

    # Then
    root = ET.fromstring(feed_xml)
    enclosure = root.find(".//enclosure")

    assert enclosure.get("url") == sample_episode_data["audio_url"]
    assert enclosure.get("length") == str(sample_episode_data["file_size"])
    assert enclosure.get("type") == sample_episode_data["mime_type"]


def test_feed_itunes_categories(test_db, sample_feed_data):
    """
    Given: A feed with multiple categories
    When: Generating the RSS feed
    Then: iTunes categories should be correctly formatted
    """
    # Given
    feed = test_db.create_feed(**sample_feed_data)

    # When
    generator = PodcastFeedGenerator(test_db)
    feed_xml = generator.generate_feed(feed.slug)

    # Then
    root = ET.fromstring(feed_xml)
    itunes_ns = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}
    category = root.find(".//itunes:category", itunes_ns)

    assert category.get("text") == "Technology"


def test_episode_guid_uniqueness(test_db, sample_feed_data, sample_episode_data):
    """
    Given: A feed with an existing episode
    When: Attempting to add another episode with the same GUID
    Then: An integrity error should be raised
    """
    # Given
    feed = test_db.create_feed(**sample_feed_data)
    test_db.add_episode(feed.slug, **sample_episode_data)

    # When/Then
    with pytest.raises(Exception):  # SQLite will raise an IntegrityError
        test_db.add_episode(feed.slug, **sample_episode_data)


def test_feed_name_uniqueness(test_db, sample_feed_data):
    """
    Given: An existing feed
    When: Attempting to create another feed with the same name
    Then: An integrity error should be raised
    """
    # Given
    test_db.create_feed(**sample_feed_data)

    # When/Then
    with pytest.raises(Exception):  # SQLite will raise an IntegrityError
        test_db.create_feed(**sample_feed_data)
