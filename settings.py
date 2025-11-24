import sys
from os import path
from pathlib import Path

from dotenv import load_dotenv
from environs import Env
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

# Initialize env
env = Env()


def load_env_file():
    # First try to load from environment variables directly
    load_dotenv()

    # If running from PyInstaller bundle
    if getattr(sys, "frozen", False):
        # Try to find .env file in the same directory as the executable
        app_dir = Path(sys.executable).parent
        dotenv_path = app_dir / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path)
        else:
            print(f"No .env file found at: {dotenv_path}")
            print("Please place your .env file in the same directory as the executable")
    else:
        # Development environment - load from project root
        dotenv_path = Path(__file__).parent / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path)


# Load environment variables BEFORE accessing them
load_env_file()

LANGSMITH_TRACING = env.bool("LANGSMITH_TRACING", False)
LANGSMITH_ENDPOINT = env.str("LANGSMITH_ENDPOINT", "")
LANGSMITH_API_KEY = env.str("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = env.str("LANGSMITH_PROJECT", "")

PAI_WORKSPACE = env.str("PAI_WORKSPACE", "/Users/scantist/Projects/playwright-demo")
OPENAI_API_KEY = env.str(
    "OPENAI_API_KEY",
)
PAI_THINKING_MODEL = init_chat_model(
    env.str("PAI_THINKING_MODEL", "gpt-5.1"), max_retries=3
)

COORDINATION_MODEL = init_chat_model(
    env.str("COORDINATION_MODEL", "gpt-5"), max_retries=3
)
EXECUTOR_MODEL = init_chat_model(env.str("EXECUTOR_MODEL", "gpt-4o"), max_retries=3)
SUMMARIZATION_MODEL = init_chat_model(
    env.str("SUMMARIZATION_MODEL", "gpt-4.1"), max_retries=3
)
SUMMARIZATION_MAX_TOKEN = env.int("SUMMARIZATION_MAX_TOKEN", 100000)
SUMMARIZATION_MESSAGES_TO_KEEP = env.int("SUMMARIZATION_MESSAGES_TO_KEEP", 1)
EXECUTOR_MESSAGE_WORKSPACE = path.join(PAI_WORKSPACE, "executor_logs")
PYTHON_TOOL_WORKSPACE = path.join(PAI_WORKSPACE, "python")
TERMINAL_TOOL_WORKSPACE = path.join(PAI_WORKSPACE, "terminal")

for _path in [
    PAI_WORKSPACE,
    PYTHON_TOOL_WORKSPACE,
    TERMINAL_TOOL_WORKSPACE,
    EXECUTOR_MESSAGE_WORKSPACE,
]:
    Path(_path).mkdir(parents=True, exist_ok=True)
