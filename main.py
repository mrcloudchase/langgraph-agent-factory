"""Agent Factory demo — each agentic system defined in a single YAML file.

    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
    python main.py
"""

from __future__ import annotations

import os
import sys

from engine import AgentFactory, ToolRegistry
from engine.tools import BUILTIN_TOOLS


def hr(label: str) -> None:
    pad = (62 - len(label) - 2) // 2
    print(f"\n{'─' * pad} {label} {'─' * pad}\n")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY before running.")

    tools = ToolRegistry()
    tools.register(*BUILTIN_TOOLS)

    factory = AgentFactory(tools)
    factory.load("agents/")

    Q = "What is LangGraph and what are its main use cases?"

    # ── react ─────────────────────────────────────────────────────────────────
    hr("react  —  agents/web-researcher.yaml")
    result = factory.build("web-researcher").invoke(
        {"messages": [{"role": "user", "content": Q}]}
    )
    print(result["messages"][-1].content)

    # ── chain ─────────────────────────────────────────────────────────────────
    hr("chain  —  agents/research-pipeline.yaml")
    result = factory.build("research-pipeline").invoke(
        {"messages": [{"role": "user", "content": Q}]}
    )
    print(result["messages"][-1].content)

    # ── router ────────────────────────────────────────────────────────────────
    hr("router  —  agents/smart-router.yaml")
    result = factory.build("smart-router").invoke(
        {"messages": [{"role": "user", "content": "Calculate the 20th Fibonacci number."}]}
    )
    print(result["messages"][-1].content)

    # ── parallel ──────────────────────────────────────────────────────────────
    hr("parallel  —  agents/parallel-perspectives.yaml")
    result = factory.build("parallel-perspectives").invoke(
        {"messages": [{"role": "user", "content": Q}]}
    )
    print(result["messages"][-1].content)

    # ── evaluator ─────────────────────────────────────────────────────────────
    hr("evaluator  —  agents/write-and-review.yaml")
    result = factory.build("write-and-review").invoke(
        {"messages": [{"role": "user", "content": "Explain LangGraph in two sentences."}]}
    )
    print(result["messages"][-1].content)

    # ── orchestrator ──────────────────────────────────────────────────────────
    hr("orchestrator  —  agents/research-team.yaml")
    try:
        result = factory.build("research-team").invoke(
            {"messages": [{"role": "user", "content": Q}]}
        )
        print(result["messages"][-1].content)
    except ImportError as exc:
        print(f"Skipped — {exc}")


if __name__ == "__main__":
    main()
