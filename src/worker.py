from celery import Celery
import asyncio

import services
import models
import config

celery_app = Celery(
    __name__,
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND
)

@celery_app.task(name="generate_project_task")
def generate_project_task(request_data):
    """
    Celery task to generate a project asynchronously.
    """
    try:
        validated_request = models.UnifiedProjectRequest.model_validate(request_data)
        result = asyncio.run(services.unified_project_generation(validated_request))
        return result.model_dump_json()
    except Exception as e:
        # It's good practice to log the error
        print(f"Task failed: {e}")
        # Celery will store the exception in the result backend
        raise
