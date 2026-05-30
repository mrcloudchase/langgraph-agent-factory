"""Agent Factory — one example per Anthropic pattern + swarm.

Anthropic "Building Effective Agents" patterns implemented:
  1. react         Single agent with tools
  2. chain         Prompt chaining (A → B → C)
  3. router        Routing (classify → one agent → done)
  4. parallel      Parallelization (all agents concurrently, merged)
  5. orchestrator  Orchestrator-subagents (requires langgraph-supervisor)
  6. evaluator     Evaluator-optimizer (generate → critique → loop)
  7. swarm         Agent handoff (requires langgraph-swarm)

    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
    python main.py
"""

from __future__ import annotations

import os
import sys

from engine import AgentFactory, AgentSpec, ToolRegistry
from engine.tools import run_python, web_fetch, web_search


def hr(label: str) -> None:
    pad = (62 - len(label) - 2) // 2
    print(f"\n{'─' * pad} {label} {'─' * pad}\n")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY before running.  See .env.example.")

    # ── Tool registry ─────────────────────────────────────────────────────────
    tools = ToolRegistry()
    tools.register(web_search, web_fetch, run_python)
    factory = AgentFactory(tools)

    # ── Define leaf agents (all react) ────────────────────────────────────────
    factory.register(AgentSpec(
        name="researcher",
        type="react",
        system_prompt="Search the web and return concise, cited findings.",
        tools=["web_search", "web_fetch"],
    ))

    factory.register(AgentSpec(
        name="summariser",
        type="react",
        system_prompt="Distill research notes into a 3-bullet executive summary. Be direct.",
        tools=[],
    ))

    factory.register(AgentSpec(
        name="critic",
        type="react",
        system_prompt=(
            "You are a strict quality reviewer. "
            "Reply ACCEPTED if the response is clear and accurate. "
            "Otherwise reply REJECTED: <specific feedback>."
        ),
        tools=[],
    ))

    factory.register(AgentSpec(
        name="coder",
        type="react",
        system_prompt="Write and run Python to answer data or calculation questions.",
        tools=["run_python"],
    ))

    factory.register(AgentSpec(
        name="writer",
        type="react",
        system_prompt="Write clear, well-structured prose answers.",
        tools=[],
    ))

    Q = "What is LangGraph and what are its main use cases?"

    # ── 1. react ──────────────────────────────────────────────────────────────
    hr("1. react — single agent with tools")
    agent = factory.build("researcher")
    result = agent.invoke({"messages": [{"role": "user", "content": Q}]})
    print(result["messages"][-1].content)

    # ── 2. chain (prompt chaining) ────────────────────────────────────────────
    hr("2. chain — researcher → summariser")
    factory.register(AgentSpec(
        name="research-then-summarise",
        type="chain",
        agents=["researcher", "summariser"],
    ))
    agent = factory.build("research-then-summarise")
    result = agent.invoke({"messages": [{"role": "user", "content": Q}]})
    print(result["messages"][-1].content)

    # ── 3. router (routing) ───────────────────────────────────────────────────
    hr("3. router — classifier picks researcher or coder")
    factory.register(AgentSpec(
        name="smart-router",
        type="router",
        agents=["researcher", "coder"],
    ))
    agent = factory.build("smart-router")
    result = agent.invoke({"messages": [{"role": "user", "content": "Calculate the 20th Fibonacci number."}]})
    print(result["messages"][-1].content)

    # ── 4. parallel (parallelization) ─────────────────────────────────────────
    hr("4. parallel — researcher + coder concurrently")
    factory.register(AgentSpec(
        name="parallel-team",
        type="parallel",
        agents=["researcher", "coder"],
    ))
    agent = factory.build("parallel-team")
    result = agent.invoke({"messages": [{"role": "user", "content": Q}]})
    print(result["messages"][-1].content)

    # ── 5. orchestrator ───────────────────────────────────────────────────────
    hr("5. orchestrator — plans across researcher + coder + writer")
    try:
        factory.register(AgentSpec(
            name="orchestrated-team",
            type="orchestrator",
            system_prompt=(
                "You coordinate a research team. Use researcher for web lookups, "
                "coder for calculations, and writer to produce the final answer."
            ),
            agents=["researcher", "coder", "writer"],
        ))
        agent = factory.build("orchestrated-team")
        result = agent.invoke({"messages": [{"role": "user", "content": Q}]})
        print(result["messages"][-1].content)
    except ImportError as exc:
        print(f"Skipped — {exc}")

    # ── 6. evaluator (evaluator-optimizer) ───────────────────────────────────
    hr("6. evaluator — writer generates, critic reviews, loops until accepted")
    factory.register(AgentSpec(
        name="write-and-review",
        type="evaluator",
        agents=["writer", "critic"],
        max_iterations=3,
    ))
    agent = factory.build("write-and-review")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Explain what a LangGraph StateGraph is in two sentences."}],
    })
    print(result["messages"][-1].content)

    # ── 7. swarm ──────────────────────────────────────────────────────────────
    hr("7. swarm — researcher and summariser hand off freely")
    try:
        factory.register(AgentSpec(
            name="research-swarm",
            type="swarm",
            agents=["researcher", "summariser"],
        ))
        agent = factory.build("research-swarm")
        result = agent.invoke({"messages": [{"role": "user", "content": Q}]})
        print(result["messages"][-1].content)
    except ImportError as exc:
        print(f"Skipped — {exc}")


if __name__ == "__main__":
    main()
