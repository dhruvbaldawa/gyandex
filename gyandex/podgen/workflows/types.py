from typing import List

from pydantic import BaseModel, Field


class ContentAnalysis(BaseModel):
    complexity: int = Field(description="Content complexity on a scale of 1-5 (1=very simple, 5=highly complex)")
    density: int = Field(description="Information density on a scale of 1-5 (1=sparse, 5=extremely dense)")
    concept_count: int = Field(description="Estimate of how many distinct key concepts are presented")
    topic_breadth: int = Field(description="Number of separate topics or themes covered")
    optimal_segments: int = Field(description="Recommended number of podcast segments (3-8)")
    explanation: str = Field(description="Brief explanation of the reasoning behind the analysis")


class OutlineSegment(BaseModel):
    name: str = Field(description="Name of the podcast segment")
    duration: int = Field(description="Duration of segment in minutes")
    talking_points: List[str] = Field(description="Key points to cover in this segment")
    transition: str = Field(
        description="Transition text to the next segment. Use 'Outro, and wrap up the podcast' "
        "if there is no transition",
        default="",
    )


class PodcastOutline(BaseModel):
    title: str = Field(description="Title of the podcast episode")
    description: str = Field(
        description="A short description of the podcast episode that provides a brief overview to the listener."
    )
    total_duration: int = Field(description="Total podcast duration in minutes")
    segments: List[OutlineSegment] = Field(description="List of podcast segments")


class DialogueLine(BaseModel):
    speaker: str
    text: str = Field(
        description="Dialogue text, ONLY use plain-text without any formatting, use quotes."
        "DO NOT USE ASTERISKS OR UNDERSCORES FOR EMPHASIS! I REPEAT DO NOT USE THEM!"
    )


class ScriptSegment(BaseModel):
    name: str
    duration: int = Field(description="Duration of the script in minutes")
    dialogue: List[DialogueLine]


class PodcastEpisode(BaseModel):
    title: str
    description: str
    dialogues: List[DialogueLine]
