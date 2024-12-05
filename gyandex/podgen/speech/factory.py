from typing import Union

from ..config.schema import GoogleCloudTTSConfig
from .google_cloud import GoogleTTSEngine


# @TODO: Centralize this type and move this to a common place
def get_text_to_speech_engine(tts_config: Union[GoogleCloudTTSConfig]):
    if tts_config.provider == "google-cloud":
        return GoogleTTSEngine(tts_config.participants)
    else:
        raise NotImplementedError(f"Unsupported TTS provider: {tts_config.provider}")
