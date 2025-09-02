from __future__ import annotations
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MODEL = os.getenv("ASSISTANT_MODEL", "gpt-4o-mini")


def get_llm() -> ChatOpenAI:
    # temperature low for tool-calling determinism
    return ChatOpenAI(model=MODEL, temperature=0)
