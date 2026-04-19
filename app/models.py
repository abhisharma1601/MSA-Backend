"""
Data models for the Coding Agent API
"""

from pydantic import BaseModel
from typing import Optional


class PromptRequest(BaseModel):
    """Request model for sending prompts to the agent"""
    prompt: str
    chat_id: Optional[int] = None


class GenerateApiKeyRequest(BaseModel):
    """Request model for generating API keys"""
    user_id: int
    expiry_days: Optional[int] = 2
