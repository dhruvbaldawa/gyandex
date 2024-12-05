from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, HttpUrl


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


class AlexandriaWorkflowConfig(BaseModel):
    name: Literal["alexandria"]
    outline: Union[GoogleGenerativeAILLMConfig]
    script: Union[GoogleGenerativeAILLMConfig]
    verbose: Optional[bool] = False


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non-binary"


class Participant(BaseModel):
    name: str
    voice: str
    gender: Gender
    personality: Optional[str] = ""
    language_code: Optional[str] = "en-US"


class GoogleCloudTTSConfig(BaseModel):
    provider: Literal["google-cloud"]
    participants: List[Participant]


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
    workflow: Union[AlexandriaWorkflowConfig] = Field(discriminator="name")
    tts: Union[GoogleCloudTTSConfig] = Field(discriminator="provider")
    storage: Union[S3StorageConfig] = Field(discriminator="provider")
    feed: FeedConfig
