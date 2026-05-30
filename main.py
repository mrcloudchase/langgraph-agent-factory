"""Agent Factory demo.

Shows how to:
  1. Set up a ToolRegistry with built-in and custom tools
  2. Define agents with AgentSpec
  3. Build and run them through AgentFactory

    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
    python main.py
"""

from __future__ import annotations

import os
import sys

from engine import AgentFactory, AgentSpec, ToolRegistry
from engine.tools import run_python, web_fetch, web_search


def section(label: str) -> None:
    pad = (60 - len(label) - 2) // 2
    print(f"\n{'─' * pad} {label} {'─' * pad}\n")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY before running.  See .env.example.")

    # ── 1. Tool registry ──────────────────────────────────────────────────────
    tools = ToolRegistry()
    tools.register(web_search, web_fetch, run_python)

    @tools.tool
    def word_count(text: str) -> str:
        """Count the number of words in a block of text."""
        n = len(text.split())
        return f"{n} words"

    section("REGISTERED TOOLS")
    print("  " + ", ".join(tools.names))

    # ── 2. Agent definitions ──────────────────────────────────────────────────
    factory = AgentFactory(tools)

    factory.register(AgentSpec(
        name="researcher",
        type="react",
        system_prompt=(
            "You are a concise research assistant. "
            "Search the web and summarize findings clearly in plain language. "
            "Always cite your sources."
        ),
        tools=["web_search", "web_fetch"],
    ))

    factory.register(AgentSpec(
        name="analyst",
        type="react",
        system_prompt=(
            "You are a data analyst. "
            "Use Python to process data, run calculations, and produce clear summaries."
        ),
        tools=["run_python", "word_count"],
    ))

    section("REGISTERED AGENTS")
    print("  researcher  — web_search, web_fetch")
    print("  analyst     — run_python, word_count")

    # ── 3. Run: researcher ────────────────────────────────────────────────────
    section("RESEARCHER AGENT")
    print("Question: What is LangGraph and what problems does it solve?\n")

    researcher = factory.build("researcher")
    result = researcher.invoke({
        "messages": [{"role": "user", "content": "What is LangGraph and what problems does it solve?"}]
    })
    print(result["messages"][-1].content)

    # ── 4. Run: analyst ───────────────────────────────────────────────────────
    section("ANALYST AGENT")
    print("Question: Calculate compound interest on $10,000 at 7% for 10 years.\n")

    analyst = factory.build("analyst")
    result = analyst.invoke({
        "messages": [{
            "role": "user",
            "content": "Calculate compound interest on $10,000 at 7% annually for 10 years. Show year-by-year breakdown."
        }]
    })
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
