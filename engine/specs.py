from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AgentSpec(BaseModel):
    name: str

    # Anthropic "Building Effective Agents" patterns
    # + swarm (LangGraph multi-agent handoff)
    # sequential / supervisor kept as aliases for backward compat
    type: Literal[
        "react",          # single agent with tools
        "chain",          # prompt chaining   (sequential alias)
        "router",         # routing
        "parallel",       # parallelization
        "orchestrator",   # orchestrator-subagents  (supervisor alias)
        "evaluator",      # evaluator-optimizer
        "swarm",          # LangGraph agent handoff
        "sequential",     # alias → chain
        "supervisor",     # alias → orchestrator
    ] = "react"

    model: str = "claude-opus-4-8"
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)

    # subagent names:
    #   chain/sequential  — ordered pipeline
    #   router            — candidates (classifier picks one)
    #   parallel          — all run concurrently
    #   orchestrator      — pool the orchestrator draws from
    #   evaluator         — [0] generator, [1] evaluator
    #   swarm             — all participants
    agents: list[str] = Field(default_factory=list)

    max_iterations: int = Field(default=5, ge=1)  # evaluator-optimizer loop cap
    checkpointer: bool = False
