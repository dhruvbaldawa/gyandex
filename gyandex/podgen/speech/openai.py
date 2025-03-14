from typing import Any, Dict, List, Optional

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ..config.schema import Participant
from .base import BaseTTSEngine


class OpenAITTSEngine(BaseTTSEngine):
    def __init__(
        self,
        participants: List[Participant],
        model: str = "tts-1",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        # Call the parent class constructor to set up common functionality
        super().__init__(participants)

    def generate_voice_profile(self, participants: List[Participant]) -> Dict[str, Any]:
        """
        Maps each participant to an appropriate OpenAI voice.
        OpenAI offers voices like: alloy, echo, fable, onyx, nova, and shimmer
        """
        return {participant.name: participant.voice for participant in participants}

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(30),
        # Retry on rate limit or server errors
        retry=retry_if_exception_type((Exception)),
        reraise=True,
    )
    def synthesize_speech(self, text: str, speaker: str) -> bytes:
        """
        Synthesize speech using OpenAI's text-to-speech API.
        """
        voice = self.voices[speaker]
        speed = self.participants[speaker].speed or 1.0
        response = self.client.audio.speech.create(model=self.model, voice=voice, input=text, speed=speed)

        # OpenAI returns a stream of bytes that we can convert to our needed format
        return response.content
