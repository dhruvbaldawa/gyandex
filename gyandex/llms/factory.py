import logging
from datetime import datetime
from typing import Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_google_genai import GoogleGenerativeAI

from ..podgen.config.schema import GoogleGenerativeAILLMConfig  # @TODO: Pull this out of podgen


class LLMLoggingCallback(BaseCallbackHandler):
    def __init__(self, log_dir="assets"):
        logger = logging.getLogger("llm_logger")
        logger.setLevel(logging.INFO)

        # Create file handler with timestamp in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(f"{log_dir}/llm_logs_{timestamp}.log")
        fh.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        fh.setFormatter(formatter)

        logger.addHandler(fh)
        self.logger = logger

    def on_llm_start(self, serialized, prompts, **kwargs):
        # Log each prompt on a new line for readability
        prompt_text = "\n".join(prompts)
        self.logger.info(f"\n=== PROMPT ===\n{prompt_text}\n")

    def on_llm_end(self, response, **kwargs):
        # Extract and format the plain text response
        response_text = response.generations[0][0].text
        self.logger.info(f"\n=== RESPONSE ===\n{response_text}\n")

    def on_llm_error(self, error, **kwargs):
        self.logger.error(f"\n=== ERROR ===\n{str(error)}\n")


# @TODO: Centralize this argument type in a single place
def get_model(config: Union[GoogleGenerativeAILLMConfig], log_dir="assets"):  # pyright: ignore [reportInvalidTypeArguments]
    if config.provider == "google-generative-ai":
        return GoogleGenerativeAI(
            model=config.model,
            temperature=config.temperature,
            google_api_key=config.google_api_key,  # pyright: ignore [reportCallIssue]
            max_output_tokens=8192,  # @TODO: Move this to config params  # pyright: ignore [reportCallIssue]
            callbacks=[LLMLoggingCallback(log_dir)],
        )
    else:
        raise NotImplementedError(f"Provider {config.provider} not implemented")
