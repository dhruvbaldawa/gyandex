from typing import Optional, Dict, Any

import requests
from pydantic import BaseModel

from ..podgen.config.schema import ContentConfig, ContentFormat  # @TODO: Pull this out of podgen


# @TODO: pull this out of this file
class Document(BaseModel):
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    content: str


def load_content(content_config: ContentConfig) -> Document:
    if content_config.format != ContentFormat.HTML:
        raise NotImplementedError(f"Unsupported content format: {content_config.format}")
    return fetch_url(content_config.source)


def fetch_url(url) -> Document:
    headers = { "Accept": "application/json" }
    response = requests.get(f"https://r.jina.ai/{url}", headers=headers)
    # @TODO: Add error handling
    content = response.json()
    return Document(title=content['data']['title'], content=content['data']['content'], metadata={
        'url': content['data']['url'],
        'description': content['data']['description'],
    })
