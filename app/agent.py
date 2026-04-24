"""
Agent code for handling Gemini API calls and streaming responses.
"""

import json
import os

import requests
from dotenv import load_dotenv


GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta"

load_dotenv()


def run_agent_stream(user_prompt: str, context_list: list = None):
    """Stream the agent response word by word"""

    if context_list is None:
        context_list = []

    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not gemini_api_key:
        yield "Error: Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable"
        return

    # Prepare the prompt for the agent
    system_prompt = """Your name is MisterChief. If someone asks your name, who you are, or what they should call you, answer that you are MisterChief.

You are an experienced software engineer and computer science assistant. You are strong across programming languages, SQL and databases, backend and frontend development, APIs, system design, operating systems, networking, debugging, testing, algorithms, data structures, performance, security basics, and practical engineering tradeoffs. You stay focused on the user's request, call out assumptions, and produce work that can be used immediately.

Handle this software work request responsibly:

Requirements:
- Use the name MisterChief when introducing yourself
- Identify the real goal before answering
- Use the language, tool, framework, or format that best matches the user's question
- Give a practical plan when implementation details are unclear
- Write clean production-ready code when code is needed
- Put code or queries inside fenced Markdown code blocks with the correct language tag, such as ```java, ```sql, ```python, ```typescript, ```bash, or ```text
- Add useful comments only where they clarify non-obvious logic
- Handle edge cases, debugging steps, and tests
- Keep the response focused and directly usable
"""

    # Build context string from context list
    context_str = ""
    if context_list:
        context_str = "\n\nPrevious Responses:\n"
        for idx, item in enumerate(context_list, 1):
            if isinstance(item, dict):
                key = list(item.keys())[0]
                context_str += f"Response {idx}: {item[key]}\n"

    full_prompt = f"{system_prompt}{context_str}\nUser Query: {user_prompt}"

    try:
        response = requests.post(
            f"{GEMINI_API_URL}/models/{gemini_model}:streamGenerateContent?alt=sse",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": gemini_api_key,
            },
            json={
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": full_prompt}],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                },
            },
            stream=True,
            timeout=(10, 300),
        )
    except requests.RequestException as exc:
        yield f"Error: Failed to connect to Gemini API - {exc}"
        return

    if response.status_code != 200:
        yield f"Error: {response.status_code} - {response.text}"
        return

    # Gemini streaming uses SSE with JSON payloads in `data:` lines.
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue

        payload = line[6:].strip()
        if not payload:
            continue

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        for candidate in data.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                token = part.get("text")
                if token:
                    yield token
