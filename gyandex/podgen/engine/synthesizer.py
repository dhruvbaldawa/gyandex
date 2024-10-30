from google.cloud import texttospeech

from gyandex.podgen.engine.workflows import PodcastSegment  # @TODO: Pull this out of workflows


class TTSEngine:
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
