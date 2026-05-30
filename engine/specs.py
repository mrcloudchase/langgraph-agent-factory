from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AgentSpec(BaseModel):
    name: str
    type: Literal[
        "react",        # ReAct agent — think → tool call → observe → repeat
        "chain",        # Prompt chaining — A → B → C, output fed forward
        "router",       # Routing — classify once, dispatch to one specialist
        "parallel",     # Parallelization — all agents concurrently, outputs merged
        "orchestrator", # Orchestrator-subagents — LLM dynamically plans & coordinates
        "evaluator",    # Evaluator-optimizer — generate → critique loop
        "swarm",        # Swarm — agents hand off to each other freely
    ] = "react"

    model: str = "claude-opus-4-8"
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)

    # Subagent names — semantics vary by type:
    #   chain        ordered pipeline [first, ..., last]
    #   router       candidates the classifier picks from
    #   parallel     all run concurrently
    #   orchestrator pool the orchestrator draws from
    #   evaluator    [generator, evaluator]
    #   swarm        all participants
    agents: list[str] = Field(default_factory=list)

    max_iterations: int = Field(default=5, ge=1)  # evaluator loop cap
    checkpointer: bool = False                     # enable memory on react agents
