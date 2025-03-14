from typing import Any, Dict, List

from google.api_core.exceptions import ResourceExhausted
from google.cloud import texttospeech
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ..config.schema import Gender, Participant
from .base import BaseTTSEngine


class GoogleTTSEngine(BaseTTSEngine):
    def __init__(self, participants: List[Participant]):
        self.client = texttospeech.TextToSpeechClient()
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3, effects_profile_id=["headphone-class-device"]
        )
        super().__init__(participants)

    def generate_voice_profile(self, participants: List[Participant]) -> Dict[str, Any]:
        def resolve_gender(gender: Gender):
            if gender == Gender.FEMALE:
                return texttospeech.SsmlVoiceGender.FEMALE
            elif gender == Gender.MALE:
                return texttospeech.SsmlVoiceGender.MALE
            else:
                return texttospeech.SsmlVoiceGender.NEUTRAL

        voices = {}
        for participant in participants:
            # Default, use the participant name
            language_code = "en-US"
            ssml_gender = resolve_gender(participant.gender)

            # The SSML voice gender values can be: MALE, FEMALE, or NEUTRAL.
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code, name=participant.voice, ssml_gender=ssml_gender
            )
            voices[participant.name] = voice

        return voices

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(30),
        retry=retry_if_exception_type(ResourceExhausted),
        reraise=True,
    )
    def synthesize_speech(self, text: str, speaker: str) -> bytes:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        response = self.client.synthesize_speech(
            input=synthesis_input, voice=self.voices[speaker], audio_config=self.audio_config
        )
        return response.audio_content
