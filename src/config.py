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
