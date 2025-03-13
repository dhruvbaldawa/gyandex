from enum import Enum
from typing import List, Literal, Optional, TypeAlias, Union

from pydantic import BaseModel, Field, HttpUrl


# @TODO: Redo this, the content format can be better structured
class ContentFormat(Enum):
    HTML = "html"
    YOUTUBE = "youtube"
    PDF = "pdf"
    TEXT = "text"


# @TODO: Pull this out of podgen
class ContentConfig(BaseModel):
    source: str
    format: ContentFormat


class LLMProviders(Enum):
    GOOGLE_GENERATIVE_AI = "google-generative-ai"
    OPENAI = "openai"


class GoogleGenerativeAILLMConfig(BaseModel):
    provider: Literal["google-generative-ai"]
    model: str
    temperature: Optional[float] = 0.7
    api_key: str


class OpenAILLMConfig(BaseModel):
    provider: Literal["openai"]
    model: str
    temperature: Optional[float] = 0.7
    api_key: str
    base_url: Optional[str] = None


LLMConfig: TypeAlias = Union[GoogleGenerativeAILLMConfig, OpenAILLMConfig]


class AlexandriaWorkflowConfig(BaseModel):
    name: Literal["alexandria"]
    outline: LLMConfig = Field(discriminator="provider")
    script: LLMConfig = Field(discriminator="provider")
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
    enabled: bool = True
    participants: List[Participant]


class ZyphraTTSConfig(BaseModel):
    provider: Literal["zyphra"]
    enabled: bool = True
    api_key: str
    participants: List[Participant]

TTSConfig: TypeAlias = Union[GoogleCloudTTSConfig, ZyphraTTSConfig]

class S3StorageConfig(BaseModel):
    provider: Literal["s3"]
    enabled: bool = True
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
    workflow: Union[AlexandriaWorkflowConfig] = Field(discriminator="name")  # pyright: ignore [reportInvalidTypeArguments]
    tts: TTSConfig = Field(discriminator="provider")  # pyright: ignore [reportInvalidTypeArguments]
    storage: Union[S3StorageConfig] = Field(discriminator="provider")  # pyright: ignore [reportInvalidTypeArguments]
    feed: FeedConfig
