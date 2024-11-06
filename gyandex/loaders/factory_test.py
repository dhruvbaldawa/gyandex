import json

import pytest
import responses
from gyandex.loaders.factory import fetch_url


@responses.activate
def test_fetch_url_returns_json_response():
    """Tests that fetch_url successfully retrieves and returns JSON data from a URL"""
    # Given
    test_url = "test123"
    actual = {"data": {"title": "title", "content": "test content", "url": "url", "description": "description"}}
    responses.add(
        responses.GET,
        f"https://r.jina.ai/{test_url}",
        json=actual,
        status=200
    )

    # When
    result = fetch_url(test_url)

    # Then
    assert result.content == "test content"
    assert result.title == "title"
    assert result.metadata == { "url": "url", "description": "description" }


@responses.activate
def test_fetch_url_sends_correct_headers():
    """Tests that fetch_url sends the correct Accept header"""
    # Given
    test_url = "test123"
    expected_headers = {"Accept": "application/json"}
    responses.add(
        responses.GET,
        f"https://r.jina.ai/{test_url}",
        json={"data": {"title": "title", "content": "test content", "url": "url", "description": "description"}},
        status=200
    )

    # When
    fetch_url(test_url)

    # Then
    assert responses.calls[0].request.headers["Accept"] == "application/json"

@responses.activate
def test_fetch_url_constructs_correct_url():
    """Tests that fetch_url constructs the correct URL with the base and provided path"""
    # Given
    test_url = "test123"
    expected_url = f"https://r.jina.ai/{test_url}"
    responses.add(
        responses.GET,
        expected_url,
        json={"data": {"title": "title", "content": "test content", "url": "url", "description": "description"}},
        status=200
    )

    # When
    fetch_url(test_url)

    # Then
    assert responses.calls[0].request.url == expected_url
