"""Runtime health aggregation for the API layer."""

import logging

from repository import dataset as dataset_repository
from service import model as model_service

logger = logging.getLogger(__name__)


def get_health_status() -> dict[str, bool | str]:
    """Report whether required runtime assets are available right now."""
    try:
        dataset_repository.get_dataset()
        dataset_loaded = True
    except (OSError, ValueError) as exc:
        logger.warning("Health check found dataset unavailable: %s", type(exc).__name__)
        dataset_loaded = False

    return {
        "status": "ok" if dataset_loaded else "degraded",
        "model_available": model_service.is_model_available(),
        "dataset_loaded": dataset_loaded,
    }
