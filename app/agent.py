"""
Agent code for handling Ollama API calls and streaming responses
"""

import requests
import json


OLLAMA_URL = "http://52.5.122.5:11434"


def run_agent_stream(user_prompt: str, context_list: list = None):
    """Stream the agent response word by word"""
    
    if context_list is None:
        context_list = []
    
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
    
    # Ollama API call with streaming
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": "qwen2.5-coder:7b",
            "prompt": full_prompt,
            "stream": True,
            "temperature": 0.2
        },
        stream=True,
        timeout=(10, 300)
    )
    
    if response.status_code != 200:
        yield f"Error: {response.status_code} - {response.text}"
        return
    
    # Stream the response
    for line in response.iter_lines(chunk_size=1, decode_unicode=True):
        if line:
            try:
                data = json.loads(line)
                token = data.get('response')
                if token:
                    yield token
                if data.get('done', False):
                    break
            except json.JSONDecodeError:
                continue
