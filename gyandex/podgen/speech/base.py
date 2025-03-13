from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..config.schema import Participant
from ..workflows.types import DialogueLine


class BaseTTSProvider(ABC):
    """
    Abstract base class for Text-to-Speech providers.
    All TTS providers must implement these methods.
    """

    @abstractmethod
    def __init__(self, participants: List[Participant]):
        """
        Initialize the TTS provider with a list of participants.

        Args:
            participants: List of participants with their voice configurations
        """
        pass

    @abstractmethod
    def generate_voice_profile(self, participants: List[Participant]) -> Dict[str, Any]:
        """
        Generate voice profiles for all participants.

        Args:
            participants: List of participants with their voice configurations

        Returns:
            A dictionary mapping participant names to their voice profiles
        """
        pass

    @abstractmethod
    def process_segment(self, segment: DialogueLine) -> bytes:
        """
        Process a dialogue segment and convert it to speech.

        Args:
            segment: A dialogue line containing text and speaker information

        Returns:
            Audio data as bytes
        """
        pass

    @abstractmethod
    def synthesize_speech(self, text: str, speaker: str) -> bytes:
        """
        Synthesize speech for a given text and speaker.

        Args:
            text: The text to convert to speech
            speaker: The name of the speaker

        Returns:
            Audio data as bytes
        """
        pass

    @abstractmethod
    def generate_audio_file(
        self, audio_segments: List[bytes], podcast_path: str, options: Optional[Dict[str, Any]] = None
    ):
        """
        Generate an audio file from a list of audio segments.

        Args:
            audio_segments: List of audio data as bytes
            podcast_path: Path where the audio file will be saved
            options: Optional configuration for audio generation
        """
        pass
