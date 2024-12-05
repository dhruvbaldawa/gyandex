from typing import Union

from ..config.schema import PodcastConfig
from .alexandria import AlexandriaWorkflow


def get_workflow(config: PodcastConfig) -> Union[AlexandriaWorkflow]:  # pyright: ignore [reportInvalidTypeArguments]
    """Get workflow based on config"""
    if config.workflow.name == "alexandria":
        return AlexandriaWorkflow(config)
    else:
        raise NotImplementedError(f"Unsupported workflow: {config.workflow.name}")
