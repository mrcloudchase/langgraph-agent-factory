"""Agent Factory demo — all patterns driven by YAML agent definitions.

Agents are defined in agents/*.yaml.
Add a YAML file and it's available to factory.build() immediately.

    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
    python main.py
"""

from __future__ import annotations

import os
import sys

from engine import AgentFactory, ToolRegistry
from engine.tools import run_python, web_fetch, web_search


def hr(label: str) -> None:
    pad = (62 - len(label) - 2) // 2
    print(f"\n{'─' * pad} {label} {'─' * pad}\n")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY before running.  See .env.example.")

    # ── Register tools, load all agent definitions ────────────────────────────
    tools = ToolRegistry()
    tools.register(web_search, web_fetch, run_python)

    factory = AgentFactory(tools)
    factory.load("agents/")

    Q = "What is LangGraph and what are its main use cases?"

    # ── 1. react ──────────────────────────────────────────────────────────────
    hr("1 · react  —  agents/researcher.yaml")
    result = factory.build("researcher").invoke(
        {"messages": [{"role": "user", "content": Q}]}
    )
    print(result["messages"][-1].content)

    # ── 2. chain ──────────────────────────────────────────────────────────────
    hr("2 · chain  —  agents/research-chain.yaml")
    result = factory.build("research-chain").invoke(
        {"messages": [{"role": "user", "content": Q}]}
    )
    print(result["messages"][-1].content)

    # ── 3. router ─────────────────────────────────────────────────────────────
    hr("3 · router  —  agents/smart-router.yaml")
    result = factory.build("smart-router").invoke(
        {"messages": [{"role": "user", "content": "Calculate the 20th Fibonacci number."}]}
    )
    print(result["messages"][-1].content)

    # ── 4. parallel ───────────────────────────────────────────────────────────
    hr("4 · parallel  —  agents/parallel-team.yaml")
    result = factory.build("parallel-team").invoke(
        {"messages": [{"role": "user", "content": Q}]}
    )
    print(result["messages"][-1].content)

    # ── 5. orchestrator ───────────────────────────────────────────────────────
    hr("5 · orchestrator  —  agents/orchestrated-team.yaml")
    try:
        result = factory.build("orchestrated-team").invoke(
            {"messages": [{"role": "user", "content": Q}]}
        )
        print(result["messages"][-1].content)
    except ImportError as exc:
        print(f"Skipped — {exc}")

    # ── 6. evaluator ──────────────────────────────────────────────────────────
    hr("6 · evaluator  —  agents/write-and-review.yaml")
    result = factory.build("write-and-review").invoke(
        {"messages": [{"role": "user", "content": "Explain LangGraph StateGraph in two sentences."}]}
    )
    print(result["messages"][-1].content)

    # ── 7. swarm ──────────────────────────────────────────────────────────────
    hr("7 · swarm  —  agents/research-swarm.yaml")
    try:
        result = factory.build("research-swarm").invoke(
            {"messages": [{"role": "user", "content": Q}]}
        )
        print(result["messages"][-1].content)
    except ImportError as exc:
        print(f"Skipped — {exc}")


if __name__ == "__main__":
    main()
