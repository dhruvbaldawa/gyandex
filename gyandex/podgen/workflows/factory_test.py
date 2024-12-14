import pytest

from ..config.schema import AlexandriaWorkflowConfig, PodcastConfig
from .alexandria import AlexandriaWorkflow
from .factory import get_workflow


def test_get_workflow_returns_alexandria():
    """Tests that get_workflow creates an AlexandriaWorkflow instance with correct config"""
    # Given
    workflow_config = AlexandriaWorkflowConfig.model_construct(name="alexandria")
    config = PodcastConfig.model_construct(workflow=workflow_config)

    # When
    workflow = get_workflow(config)

    # Then
    assert isinstance(workflow, AlexandriaWorkflow)


def test_get_workflow_raises_for_unsupported_workflow():
    """Tests that get_workflow raises NotImplementedError for unsupported workflows"""
    # Given
    workflow_config = AlexandriaWorkflowConfig.model_construct(name="unsupported")
    config = PodcastConfig.model_construct(workflow=workflow_config)

    # When/Then
    with pytest.raises(NotImplementedError, match="Unsupported workflow: unsupported"):
        get_workflow(config)
