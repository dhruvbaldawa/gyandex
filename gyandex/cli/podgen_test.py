from unittest.mock import Mock, patch

import pytest

from gyandex.cli.podgen import main


def test_cli_help_command():
    """Tests that help command prints help message and exits"""
    # When
    with (
        patch("argparse.ArgumentParser.parse_args", return_value=Mock(config_path="--help")),
        patch("argparse.ArgumentParser.print_help") as mock_help,
    ):
        main()

        # Then
        mock_help.assert_called_once()


def test_invalid_config_path():
    """Tests handling of invalid configuration file path"""
    # Given
    invalid_path = "nonexistent.yaml"

    # When/Then
    with (
        pytest.raises(FileNotFoundError),
        patch("argparse.ArgumentParser.parse_args", return_value=Mock(config_path=invalid_path)),
    ):
        main()
