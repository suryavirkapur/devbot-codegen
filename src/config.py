import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set.")

BASE_GENERATED_REPOS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../generated_repos")
)

DEFAULT_MODEL = "gpt-4.1"

# Celery Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL
