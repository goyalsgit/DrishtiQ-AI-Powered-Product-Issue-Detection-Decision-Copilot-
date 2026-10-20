"""Gemini configuration. Credentials stay in the process environment."""
import os


def make_llm(model_name=None, *, api_key=None):
    # An explicit session key avoids sharing browser credentials through os.environ.
    key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "Set GOOGLE_API_KEY locally, or run lesson 10 with --ask-key. "
            "Do not put your key in source code."
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model_name or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        api_key=key,
        vertexai=False,
        timeout=60,
        max_retries=1,
        max_tokens=2048,
    )
