from io import BytesIO
from typing import List, Optional, Dict, Any

from google.cloud import texttospeech
from pydub import AudioSegment

from gyandex.podgen.engine.workflows import PodcastSegment  # @TODO: Pull this out of workflows


class TTSEngine:
    # @TODO: Accept configuration to tweak the voice
    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()
        self.voices = {
            'HOST1': texttospeech.VoiceSelectionParams(
                language_code='en-US',
                name='en-US-Journey-D',
                ssml_gender=texttospeech.SsmlVoiceGender.MALE
            ),
            'HOST2': texttospeech.VoiceSelectionParams(
                language_code='en-US',
                name='en-US-Journey-O',
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
            )
        }
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            effects_profile_id=['headphone-class-device']
        )

    def process_segment(self, segment: PodcastSegment) -> bytes:
        # ssml = self.generate_ssml(segment)
        return self.synthesize_speech(segment.text, segment.speaker)

    def synthesize_speech(self, text: str, speaker: str) -> bytes:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=self.voices[speaker],
            audio_config=self.audio_config
        )
        return response.audio_content

    def generate_audio_file(self, audio_segments: List[bytes], podcast_path: str, options: Optional[Dict[str, Any]] = None):
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
                combined = combined.append(segment_audio, crossfade=options['crossfade'])
            else:
                combined += segment_audio
            previous_segment = segment

        # Save final podcast
        combined.export(podcast_path, format="mp3")