from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AgentSpec(BaseModel):
    name: str
    type: Literal[
        "llm",          # single LLM call — one invocation, returns, done
        "react",        # ReAct agent — LLM controls a tool-call loop
        "chain",        # workflow: A → B → C, output fed forward
        "router",       # workflow: classify once, dispatch to one step
        "parallel",     # workflow: all steps run concurrently, outputs merged
        "orchestrator", # workflow: LLM supervisor coordinates steps dynamically
        "evaluator",    # workflow: generate → critique loop until accepted
    ] = "react"

    model: str = "claude-opus-4-8"
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)          # for llm / react
    steps: list[AgentSpec] = Field(default_factory=list)    # for workflow types
    max_iterations: int = Field(default=5, ge=1)            # evaluator loop cap
    checkpointer: bool = False                              # react only


AgentSpec.model_rebuild()
