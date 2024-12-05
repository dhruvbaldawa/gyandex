from typing import List

from pydantic import BaseModel, Field


class OutlineSegment(BaseModel):
    name: str = Field(description="Name of the podcast segment")
    duration: int = Field(description="Duration of segment in minutes")
    talking_points: List[str] = Field(description="Key points to cover in this segment")
    transition: str = Field(description="Transition text to the next segment. Use 'Closing remarks' if there is no transition", default="")

class PodcastOutline(BaseModel):
    title: str = Field(description="Title of the podcast episode")
    description: str = Field(description="Description of the podcast episode")
    total_duration: int = Field(description="Total podcast duration in minutes")
    segments: List[OutlineSegment] = Field(description="List of podcast segments")

class DialogueLine(BaseModel):
    speaker: str
    text: str

class ScriptSegment(BaseModel):
    name: str
    duration: int = Field(description="Duration of the script in minutes")
    dialogue: List[DialogueLine]

class PodcastEpisode(BaseModel):
    title: str
    description: str
    dialogues: List[DialogueLine]
