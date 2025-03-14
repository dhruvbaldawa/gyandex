from io import BytesIO
from typing import Any, Dict, List, Optional

import zyphra
from pydub import AudioSegment

from ..config.schema import Gender, Participant
from ..workflows.types import DialogueLine
from .base import BaseTTSProvider


class ZyphraTTSEngine(BaseTTSProvider):
    """
    Zyphra Text-to-Speech implementation.
    """

    def __init__(self, participants: List[Participant], api_key: str):
        """
        Initialize the Zyphra TTS engine.

        Args:
            participants: List of participants with their voice configurations
            api_key: Zyphra API key for authentication
        """
        self.client = zyphra.ZyphraClient(api_key=api_key)
        self.voices = self.generate_voice_profile(participants)
        self.api_key = api_key

    def generate_voice_profile(self, participants: List[Participant]) -> Dict[str, Any]:
        """
        Generate voice profiles for all participants using Zyphra TTS.

        Args:
            participants: List of participants with their voice configurations

        Returns:
            A dictionary mapping participant names to their Zyphra voice profiles
        """
        def resolve_gender(gender: Gender) -> str:
            if gender == Gender.FEMALE:
                return "female"
            elif gender == Gender.MALE:
                return "male"
            return "neutral"

        return {
            participant.name: {
                "voice": participant.voice,
                "gender": resolve_gender(participant.gender),
                "language_code": participant.language_code or "en-us"
            }
            for participant in participants
        }

    def process_segment(self, segment: DialogueLine) -> bytes:
        """
        Process a dialogue segment and convert it to speech using Zyphra TTS.

        Args:
            segment: A dialogue line containing text and speaker information

        Returns:
            Audio data as bytes
        """
        return self.synthesize_speech(segment.text, segment.speaker)

    def synthesize_speech(self, text: str, speaker: str) -> bytes:
        """
        Synthesize speech for a given text and speaker using Zyphra TTS.

        Args:
            text: The text to convert to speech
            speaker: The name of the speaker

        Returns:
            Audio data as bytes
        """
        voice_profile = self.voices[speaker]

        # Call Zyphra API to generate speech
        audio_data = self.client.audio.speech.create(
            text=text,
            voice=voice_profile["voice"],
            language_code=voice_profile["language_code"],
            model="zonos-v0.1-hybrid",
            mime_type="audio/mp3",
        )

        # Ensure we always return bytes
        if isinstance(audio_data, bytes):
            return audio_data
        else:
            # If audio_data is a Path object, read the file and return its contents
            from pathlib import Path
            if isinstance(audio_data, Path):
                return audio_data.read_bytes()

            # If it's a string path, convert it to Path first
            if isinstance(audio_data, str):
                return Path(audio_data).read_bytes()

            # As a fallback, try to convert whatever we got to bytes
            return bytes(audio_data)

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
        if options is None:
            options = {
                "crossfade": 200,
            }

        combined = AudioSegment.empty()
        previous_segment = None
        for segment in audio_segments:
            segment_audio = AudioSegment.from_mp3(BytesIO(segment))
            if previous_segment:
                combined = combined.append(segment_audio, crossfade=options["crossfade"])
            else:
                combined += segment_audio
            previous_segment = segment

        # Save final podcast
        combined.export(podcast_path, format="mp3")
