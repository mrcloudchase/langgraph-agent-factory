"""Agent Factory — one working example per pattern.

Patterns from Anthropic "Building Effective Agents":
  react         Single agent with tools
  chain         Prompt chaining (researcher → summariser)
  router        Routing (classifier picks researcher or coder)
  parallel      Parallelization (researcher + coder concurrently)
  orchestrator  Orchestrator-subagents  [requires langgraph-supervisor]
  evaluator     Evaluator-optimizer (writer → critic loop)
  swarm         Agent handoff          [requires langgraph-swarm]

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

    # ── Tools ─────────────────────────────────────────────────────────────────
    tools = ToolRegistry()
    tools.register(web_search, web_fetch, run_python)

    # ── Leaf agents (all react) ───────────────────────────────────────────────
    factory = AgentFactory(tools)

    factory.register(AgentSpec(
        name="researcher",
        type="react",
        system_prompt="Search the web and return concise, cited findings.",
        tools=["web_search", "web_fetch"],
    ))
    factory.register(AgentSpec(
        name="summariser",
        type="react",
        system_prompt="Distill the input into a 3-bullet executive summary. Be direct.",
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

    Q = "What is LangGraph and what are its main use cases?"

    # ── 1. react ──────────────────────────────────────────────────────────────
    hr("1 · react")
    result = factory.build("researcher").invoke(
        {"messages": [{"role": "user", "content": Q}]}
    )
    print(result["messages"][-1].content)

    # ── 2. chain ──────────────────────────────────────────────────────────────
    hr("2 · chain  —  researcher → summariser")
    factory.register(AgentSpec(
        name="research-chain", type="chain",
        agents=["researcher", "summariser"],
    ))
    result = factory.build("research-chain").invoke(
        {"messages": [{"role": "user", "content": Q}]}
    )
    print(result["messages"][-1].content)

    # ── 3. router ─────────────────────────────────────────────────────────────
    hr("3 · router  —  researcher | coder")
    factory.register(AgentSpec(
        name="smart-router", type="router",
        agents=["researcher", "coder"],
    ))
    result = factory.build("smart-router").invoke(
        {"messages": [{"role": "user", "content": "Calculate the 20th Fibonacci number."}]}
    )
    print(result["messages"][-1].content)

    # ── 4. parallel ───────────────────────────────────────────────────────────
    hr("4 · parallel  —  researcher ‖ coder")
    factory.register(AgentSpec(
        name="parallel-team", type="parallel",
        agents=["researcher", "coder"],
    ))
    result = factory.build("parallel-team").invoke(
        {"messages": [{"role": "user", "content": Q}]}
    )
    print(result["messages"][-1].content)

    # ── 5. orchestrator ───────────────────────────────────────────────────────
    hr("5 · orchestrator  —  researcher + coder + writer")
    try:
        factory.register(AgentSpec(
            name="orchestrated-team", type="orchestrator",
            system_prompt=(
                "Coordinate the team. Use researcher for web lookups, "
                "coder for calculations, writer for the final answer."
            ),
            agents=["researcher", "coder", "writer"],
        ))
        result = factory.build("orchestrated-team").invoke(
            {"messages": [{"role": "user", "content": Q}]}
        )
        print(result["messages"][-1].content)
    except ImportError as exc:
        print(f"Skipped — {exc}")

    # ── 6. evaluator ──────────────────────────────────────────────────────────
    hr("6 · evaluator  —  writer → critic loop")
    factory.register(AgentSpec(
        name="write-and-review", type="evaluator",
        agents=["writer", "critic"],
        max_iterations=3,
    ))
    result = factory.build("write-and-review").invoke(
        {"messages": [{"role": "user", "content": "Explain LangGraph StateGraph in two sentences."}]}
    )
    print(result["messages"][-1].content)

    # ── 7. swarm ──────────────────────────────────────────────────────────────
    hr("7 · swarm  —  researcher ↔ summariser")
    try:
        factory.register(AgentSpec(
            name="research-swarm", type="swarm",
            agents=["researcher", "summariser"],
        ))
        result = factory.build("research-swarm").invoke(
            {"messages": [{"role": "user", "content": Q}]}
        )
        print(result["messages"][-1].content)
    except ImportError as exc:
        print(f"Skipped — {exc}")


if __name__ == "__main__":
    main()
