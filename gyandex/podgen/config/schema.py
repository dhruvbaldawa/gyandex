from enum import Enum
from typing import Dict, List, Optional, Union

from langchain_google_genai import GoogleGenerativeAI
from pydantic import BaseModel, HttpUrl
from pydantic.v1 import validator


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


class GoogleGenerativeAIParams(BaseModel):
    google_api_key: str


# @TODO: Pull this out of podgen
class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float
    llm_params: Union[GoogleGenerativeAIParams]

    # Add validation to ensure params match provider
    @validator('llm_params')
    def validate_llm_params(cls, v, values):
        provider_param_map = {
            'google-generative-ai': GoogleGenerativeAIParams,
        }
        expected_type = provider_param_map.get(values['provider'])
        if not isinstance(v, expected_type):
            raise ValueError(f'Provider {values["provider"]} requires params of type {expected_type.__name__}')
        return v


class VoiceProfile(BaseModel):
    voice_id: str
    speaking_rate: float
    pitch: int


class TTSConfig(BaseModel):
    provider: str
    default_voice: str
    voices: Dict[str, VoiceProfile]


class StorageConfig(BaseModel):
    provider: str
    bucket: str
    region: str
    path_template: str


class FeedConfig(BaseModel):
    title: str
    description: str
    author: str
    email: str
    language: str
    categories: List[str]
    image: HttpUrl
    explicit: bool


class Segment(BaseModel):
    speaker: str
    text: str


class ContentStructure(BaseModel):
    type: str
    speaker: str


class EpisodeConfig(BaseModel):
    title: str
    content_mode: str


class PodcastConfig(BaseModel):
    version: str
    content: ContentConfig
    llm: LLMConfig  # @TODO: Rethink this because I would like to use multiple LLMs for optimizing costs
    tts: TTSConfig
    storage: StorageConfig
    feed: FeedConfig
    episode: EpisodeConfig
