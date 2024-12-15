import os

import pytest

from .loader import load_config, resolve_env_vars, resolve_nested_env_vars
from .schema import PodcastConfig


def test_resolve_env_vars_replaces_single_variable():
    """Test that resolve_env_vars replaces a single environment variable in a string"""
    # Given
    os.environ["TEST_VAR"] = "test_value"
    input_string = "prefix_${TEST_VAR}_suffix"

    # When
    result = resolve_env_vars(input_string)

    # Then
    assert result == "prefix_test_value_suffix"


def test_resolve_env_vars_handles_multiple_variables():
    """Test that resolve_env_vars replaces multiple environment variables in a string"""
    # Given
    os.environ["VAR1"] = "first"
    os.environ["VAR2"] = "second"
    input_string = "${VAR1}_middle_${VAR2}"

    # When
    result = resolve_env_vars(input_string)

    # Then
    assert result == "first_middle_second"


def test_resolve_env_vars_raises_on_missing_variable():
    """Test that resolve_env_vars raises ValueError when environment variable is not found"""
    # Given
    input_string = "${NONEXISTENT_VAR}"

    # When/Then
    with pytest.raises(ValueError, match="Environment variable NONEXISTENT_VAR not found"):
        resolve_env_vars(input_string)


def test_resolve_nested_env_vars_handles_dict():
    """Test that resolve_nested_env_vars resolves variables in nested dictionary"""
    # Given
    os.environ["NESTED_VAR"] = "value"
    input_dict = {"key1": "${NESTED_VAR}", "key2": {"nested_key": "${NESTED_VAR}"}}

    # When
    result = resolve_nested_env_vars(input_dict)

    # Then
    assert result == {"key1": "value", "key2": {"nested_key": "value"}}


def test_load_config_parses_yaml_with_env_vars(tmp_path):
    """Test that load_config properly loads YAML and resolves environment variables"""
    # Given
    os.environ["FEED_TITLE"] = "My Podcast"
    config_content = """
    version: "1.0"
    feeds:
      test_feed:
        title: ${FEED_TITLE}
        description: Test Description
    content:
      format: html
      source: https://example.com/feed
    workflow:
      name: alexandria
      verbose: true
      outline:
        provider: "google-generative-ai"
        model: "gemini-1.5-pro"
        temperature: 0.4
        api_key: "xxx"
      script:
        provider: "google-generative-ai"
        model: "gemini-1.5-flash"
        temperature: 0.8
        api_key: "xxx"
    tts:
      provider: "google-cloud"
      participants:
        - name: Sarah
          personality: |
            An enthusiastic and knowledgeable tech journalist with 10 years of experience.
            Style: Articulate, engaging, asks insightful questions, and guides the conversation smoothly.
          voice: en-US-Journey-F
          language_code: en-US
          gender: female
        - name: Mike
          personality: |
            A practical industry expert with hands-on experience.
            Style: Down-to-earth, provides real-world examples, occasionally humorous, 
            and good at breaking down complex topics.
          voice: en-US-Journey-D
          language_code: en-US
          gender: male
    storage:
      provider: s3
      bucket: test_bucket
      region: test_region
      path_template: test_path_template
      access_key: test_access_key
      secret_key: test_secret_key
    feed:
      title: My Podcast
      slug: my-podcast
      description: Test Description
      author: Test Author
      email: test@example.com
      language: en-US
      image: https://example.com/image.jpg
      website: https://example.com
      explicit: false
      categories:
        - test_category
    episode:
      title: Test Episode
      description: Test Episode Description
      author: Test Author
      url: https://example.com/episode
      published: 2023-01-01
      content_mode: auto
      auto_content:
        provider: test
        structure:
          - type: foo
            speaker: bar
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    # When
    config = load_config(str(config_file))

    # Then
    assert isinstance(config, PodcastConfig)
    assert config.feed.title == "My Podcast"
