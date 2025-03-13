from typing import Union

from ..config.schema import TTSConfig
from .base import BaseTTSProvider
from .google_cloud import GoogleTTSEngine
from .zyphra import ZyphraTTSEngine


# @TODO: Centralize this type and move this to a common place
def get_text_to_speech_engine(tts_config: TTSConfig) -> BaseTTSProvider:
    if tts_config.provider == "google-cloud":
        return GoogleTTSEngine(tts_config.participants)
    elif tts_config.provider == "zyphra":
        return ZyphraTTSEngine(tts_config.participants, tts_config.api_key)
    else:
        raise NotImplementedError(f"Unsupported TTS provider: {tts_config.provider}")
