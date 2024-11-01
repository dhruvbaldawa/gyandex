from enum import Enum
from typing import Dict, List, Optional, Union, Literal

from pydantic import BaseModel, HttpUrl, Field


# @TODO: Redo this, the content format can be better structured
class ContentFormat(Enum):
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"
    TEXT = "text"


# @TODO: Pull this out of podgen
class ContentConfig(BaseModel):
    source: str
    format: ContentFormat


class LLMProviders(Enum):
    GOOGLE_GENERATIVE_AI = "google-generative-ai"


class GoogleGenerativeAILLMConfig(BaseModel):
    provider: Literal["google-generative-ai"]
    model: str
    temperature: Optional[float] = 1.0
    google_api_key: str


class VoiceProfile(BaseModel):
    voice_id: str
    speaking_rate: float
    pitch: int


class TTSConfig(BaseModel):
    provider: str
    default_voice: str
    voices: Dict[str, VoiceProfile]


class S3StorageConfig(BaseModel):
    provider: Literal["s3"]
    bucket: str
    access_key: str
    secret_key: str
    region: Optional[str] = None
    endpoint: Optional[str] = None
    custom_domain: Optional[str] = None


class FeedConfig(BaseModel):
    title: str
    slug: str
    description: str
    author: str
    email: str
    language: str
    categories: List[str]
    image: HttpUrl
    website: HttpUrl


class Segment(BaseModel):
    speaker: str
    text: str


class ContentStructure(BaseModel):
    type: str
    speaker: str


class PodcastConfig(BaseModel):
    version: str
    content: ContentConfig
    # @TODO: Rethink this because I would like to use multiple LLMs for optimizing costs
    llm: Union[GoogleGenerativeAILLMConfig] = Field(discriminator="provider")
    tts: TTSConfig
    storage: Union[S3StorageConfig] = Field(discriminator="provider")
    feed: FeedConfig
