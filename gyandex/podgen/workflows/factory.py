from typing import Union

from .alexandria import AlexandriaWorkflow
from ..config.schema import PodcastConfig


def get_workflow(config: PodcastConfig) -> Union[AlexandriaWorkflow]:
    """Get workflow based on config"""
    if config.workflow.name == "alexandria":
        return AlexandriaWorkflow(config)
    else:
        raise NotImplementedError(f"Unsupported workflow: {config.workflow.name}")
