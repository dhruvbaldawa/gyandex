import requests

from ..podgen.config.schema import ContentConfig, ContentFormat  # @TODO: Pull this out of podgen
from .types import Document
from .youtube import fetch_youtube


def load_content(content_config: ContentConfig) -> Document:
    if content_config.format == ContentFormat.HTML:
        return fetch_url(content_config.source)
    elif content_config.format == ContentFormat.YOUTUBE:
        return fetch_youtube(content_config.source)
    raise NotImplementedError(f"Unsupported content format: {content_config.format}")


def fetch_url(url) -> Document:
    headers = {"Accept": "application/json"}
    response = requests.get(f"https://r.jina.ai/{url}", headers=headers)
    # @TODO: Add error handling
    content = response.json()
    return Document(
        title=content["data"]["title"],
        content=content["data"]["content"],
        metadata={
            "url": content["data"]["url"],
            "description": content["data"]["description"],
        },
    )
