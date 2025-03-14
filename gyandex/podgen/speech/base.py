import random
import re
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Dict, List, Optional

from pydub import AudioSegment

from ..config.schema import Participant
from ..workflows.types import DialogueLine


class BaseTTSEngine(ABC):
    """
    Base class for Text-to-Speech engines.

    Provides common functionality for all TTS engines such as text cleaning
    and audio file generation. Specific TTS engine implementations should
    inherit from this class and implement the abstract methods.
    """

    def __init__(self, participants: List[Participant]):
        """
        Initialize the TTS engine with a list of participants.

        Args:
            participants: List of participants with their voice configurations
        """
        self.participants = {participant.name: participant for participant in participants}
        self.voices = self.generate_voice_profile(participants)

    @abstractmethod
    def generate_voice_profile(self, participants: List[Participant]) -> Dict[str, Any]:
        """
        Generate a mapping of participant names to voice configurations.

        Each TTS provider has different voice configuration requirements,
        so this method needs to be implemented by subclasses.

        Args:
            participants: List of participants with their voice configurations

        Returns:
            A dictionary mapping participant names to provider-specific voice configurations
        """
        pass

    @abstractmethod
    def synthesize_speech(self, text: str, speaker: str) -> bytes:
        """
        Synthesize speech for the given text and speaker.

        Args:
            text: The text to be converted to speech
            speaker: The name of the speaker (must be in the voices dictionary)

        Returns:
            The audio data as bytes
        """
        pass

    def clean_text_for_tts(self, text: str) -> str:
        """
        Clean text to remove characters and patterns that TTS applications struggle with.

        This includes:
        - Removing markdown formatting characters used for emphasis (*, _, etc.)
        - Normalizing dashes and other punctuation used for speech patterns
        - Converting special characters to their spoken equivalents
        - Cleaning up formatting that disrupts natural speech

        Args:
            text: The text to be cleaned

        Returns:
            Cleaned text suitable for TTS processing
        """
        # Special treatment for compound words with hyphens
        # Store compound words with hyphens to restore them later
        compound_words = re.findall(r"([a-zA-Z]+)-([a-zA-Z]+)", text)

        # Handle the "Wait, *hold on*" pattern
        text = re.sub(r"Wait, \*hold on\*", r"Wait, hold on", text)

        # ===== Handle markdown-style formatting =====
        # Process double-character markdown first to avoid partial matches
        # Remove double asterisks for bold (**word**)
        cleaned_text = re.sub(r"\*\*([^*\n]+?)\*\*", r"\1", text)

        # Remove double underscores for bold (__word__)
        cleaned_text = re.sub(r"__([^_\n]+?)__", r"\1", cleaned_text)

        # Then handle single-character markdown
        # Remove asterisks used for emphasizing words/phrases (*word* or *multiple words*)
        cleaned_text = re.sub(r"\*([^*\n]+?)\*", r"\1", cleaned_text)

        # Remove underscores used for emphasis (_word_ or _multiple words_)
        cleaned_text = re.sub(r"_([^_\n]+?)_", r"\1", cleaned_text)

        # ===== Handle special characters and patterns =====
        # Process math expressions - convert to spoken form
        # Handle math expressions with spaces around the asterisk
        cleaned_text = re.sub(r"(\d+)\s*\*\s*(\d+)", r"\1 times \2", cleaned_text)
        # Handle math expressions without spaces (like 5*2)
        cleaned_text = re.sub(r"(\d+)\*(\d+)", r"\1 times \2", cleaned_text)

        # Fix TPS report style references (common in business discussions)
        cleaned_text = re.sub(r"([A-Z]{2,})\s+report", r"\1 report", cleaned_text)

        # If any stray formatting characters remain, remove them
        cleaned_text = cleaned_text.replace(" * ", " ").replace(" _ ", " ")

        # Normalize various dash patterns used for interruptions/breaks
        # Replace em dash or double dash patterns with a comma and space for a natural pause
        cleaned_text = cleaned_text.replace("—", ", ")
        cleaned_text = cleaned_text.replace("--", ", ")

        # Fix patterns like "—word" that indicate interruption or continuation
        # Look for dash followed immediately by a word and replace with appropriate pause
        cleaned_text = re.sub(r"[—-]([a-zA-Z])", r", \1", cleaned_text)

        # Remove repeated whitespace
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

        # Restore compound words with hyphens that might have been converted to "word, word"
        for first, second in compound_words:
            cleaned_text = cleaned_text.replace(f"{first}, {second}", f"{first}-{second}")

        return cleaned_text.strip()

    def process_segment(self, segment: DialogueLine) -> bytes:
        """
        Process a dialogue segment for TTS conversion.

        Cleans the text to remove problematic characters before sending to the TTS engine.

        Args:
            segment: The dialogue line to be processed

        Returns:
            The audio data as bytes
        """
        cleaned_text = self.clean_text_for_tts(segment.text)
        return self.synthesize_speech(cleaned_text, segment.speaker)

    def generate_audio_file(
        self, audio_segments: List[bytes], podcast_path: str, options: Optional[Dict[str, Any]] = None
    ):
        """
        Combine multiple audio segments into a single audio file and save it.

        Args:
            audio_segments: List of audio segments as bytes
            podcast_path: Path where the final audio file will be saved
            options: Optional dictionary of options for audio processing
        """
        if options is None:
            options = {
                "crossfade": [100, 300],
            }

        combined = AudioSegment.empty()
        previous_segment = None
        for segment in audio_segments:
            segment_audio = AudioSegment.from_mp3(BytesIO(segment))
            if previous_segment:
                crossfade = random.randint(options["crossfade"][0], options["crossfade"][1])
                combined = combined.append(segment_audio, crossfade=crossfade)
            else:
                combined += segment_audio
            previous_segment = segment

        # Save final podcast
        combined.export(podcast_path, format="mp3")
