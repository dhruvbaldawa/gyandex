import re
from io import BytesIO
from typing import Any, Dict, List, Optional

from google.api_core.exceptions import ResourceExhausted
from google.cloud import texttospeech
from pydub import AudioSegment
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ..config.schema import Gender, Participant
from ..workflows.types import DialogueLine  # @TODO: Pull this out of workflows
from .base import BaseTTSProvider


class GoogleTTSEngine(BaseTTSProvider):
    """
    Google Cloud Text-to-Speech implementation.
    """

    def __init__(self, participants: List[Participant]):
        """
        Initialize the Google Cloud TTS engine.

        Args:
            participants: List of participants with their voice configurations
        """
        self.client = texttospeech.TextToSpeechClient()
        self.voices = self.generate_voice_profile(participants)
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3, effects_profile_id=["headphone-class-device"]
        )

    def generate_voice_profile(self, participants: List[Participant]) -> Dict[str, Any]:
        """
        Generate voice profiles for all participants using Google Cloud TTS.

        Args:
            participants: List of participants with their voice configurations

        Returns:
            A dictionary mapping participant names to their Google Cloud voice profiles
        """
        def resolve_gender(gender: Gender):
            if gender == Gender.FEMALE:
                return texttospeech.SsmlVoiceGender.FEMALE
            elif gender == Gender.MALE:
                return texttospeech.SsmlVoiceGender.MALE
            return texttospeech.SsmlVoiceGender.NEUTRAL

        return {
            participant.name: texttospeech.VoiceSelectionParams(
                language_code=participant.language_code,
                name=participant.voice,
                ssml_gender=resolve_gender(participant.gender),
            )
            for participant in participants
        }

    def clean_text_for_tts(self, text: str) -> str:
        """
        Clean text to remove characters and patterns that TTS applications struggle with.

        This includes:
        - Removing markdown formatting characters used for emphasis (*, _, etc.)
        - Normalizing dashes and other punctuation used for speech patterns
        - Converting special characters to their spoken equivalents
        - Cleaning up formatting that disrupts natural speech
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
        """
        cleaned_text = self.clean_text_for_tts(segment.text)
        return self.synthesize_speech(cleaned_text, segment.speaker)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(30),
        retry=retry_if_exception_type(ResourceExhausted),
        reraise=True,
    )
    def synthesize_speech(self, text: str, speaker: str) -> bytes:
        """
        Synthesize speech for a given text and speaker using Google Cloud TTS.

        Args:
            text: The text to convert to speech
            speaker: The name of the speaker

        Returns:
            Audio data as bytes
        """
        synthesis_input = texttospeech.SynthesisInput(text=text)
        response = self.client.synthesize_speech(
            input=synthesis_input, voice=self.voices[speaker], audio_config=self.audio_config
        )
        return response.audio_content

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
            # @TODO: Fix this code-smell
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
