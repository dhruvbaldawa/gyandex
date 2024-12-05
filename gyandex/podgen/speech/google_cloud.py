from io import BytesIO
from typing import Any, Dict, List, Optional

from google.cloud import texttospeech
from pydub import AudioSegment

from ..config.schema import Gender, Participant
from ..workflows.types import ScriptSegment  # @TODO: Pull this out of workflows


class GoogleTTSEngine:
    def __init__(self, participants: List[Participant]):
        self.client = texttospeech.TextToSpeechClient()
        self.voices = self.generate_voice_profile(participants)
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3, effects_profile_id=["headphone-class-device"]
        )

    def generate_voice_profile(self, participants: List[Participant]) -> Dict[str, Any]:
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

    def process_segment(self, segment: ScriptSegment) -> bytes:
        return self.synthesize_speech(segment.text, segment.speaker)

    def synthesize_speech(self, text: str, speaker: str) -> bytes:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        response = self.client.synthesize_speech(
            input=synthesis_input, voice=self.voices[speaker], audio_config=self.audio_config
        )
        return response.audio_content

    def generate_audio_file(
        self, audio_segments: List[bytes], podcast_path: str, options: Optional[Dict[str, Any]] = None
    ):
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
