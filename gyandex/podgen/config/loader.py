import os
import re

import yaml

from .schema import PodcastConfig


def resolve_env_vars(value: str) -> str:
    """Resolve ${ENV_VAR} patterns in string values"""
    if not isinstance(value, str):
        return value

    pattern = r"\${([^}^{]+)}"
    matches = re.finditer(pattern, value)

    for match in matches:
        env_var = match.group(1)
        env_value = os.environ.get(env_var)
        if env_value is None:
            raise ValueError(f"Environment variable {env_var} not found")
        value = value.replace(match.group(0), env_value)

    return value


def resolve_nested_env_vars(data):
    """Recursively resolve environment variables in nested structures"""
    if isinstance(data, dict):
        return {k: resolve_nested_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [resolve_nested_env_vars(v) for v in data]
    else:
        return resolve_env_vars(data)


def load_config(config_path: str) -> PodcastConfig:
    """Load and parse YAML config with environment variable support"""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Resolve any environment variables in the config
    config_dict = resolve_nested_env_vars(config_dict)

    # Parse with Pydantic
    return PodcastConfig(**config_dict)
