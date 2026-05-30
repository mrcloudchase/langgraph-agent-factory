from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AgentSpec(BaseModel):
    name: str
    type: Literal["react", "supervisor"] = "react"
    model: str = "claude-opus-4-8"
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    agents: list[str] = Field(default_factory=list)  # subagent names (supervisor only)
    checkpointer: bool = False
