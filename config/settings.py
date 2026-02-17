import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CORESTACK_API_KEY = os.getenv("CORESTACK_API_KEY")
CORESTACK_BASE_URL = os.getenv("CORESTACK_BASE_URL")

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_BASE_URL")
