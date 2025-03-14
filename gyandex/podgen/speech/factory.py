from ..config.schema import TTSConfig
from .google_cloud import GoogleTTSEngine
from .openai import OpenAITTSEngine


# @TODO: Centralize this type and move this to a common place
def get_text_to_speech_engine(tts_config: TTSConfig):  # pyright: ignore [reportInvalidTypeArguments]
    if tts_config.provider == "google-cloud":
        return GoogleTTSEngine(tts_config.participants)
    elif tts_config.provider == "openai":
        return OpenAITTSEngine(
            participants=tts_config.participants,
            model=tts_config.model,
            api_key=tts_config.api_key,
            base_url=tts_config.base_url,
        )
    else:
        raise NotImplementedError(f"Unsupported TTS provider: {tts_config.provider}")
