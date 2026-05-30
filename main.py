"""Agent Factory — demonstrates all five agentic system types.

    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
    python main.py

Patterns shown
--------------
  react       Single agent with a tool set
  sequential  Pipeline: researcher → summariser (output fed forward)
  parallel    Two analysts run concurrently, outputs merged
  supervisor  Router delegates to specialised subagents (requires langgraph-supervisor)
  swarm       Agents hand off to each other (requires langgraph-swarm)
  graph       Custom StateGraph registered directly
"""

from __future__ import annotations

import os
import sys

from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages

from engine import AgentFactory, AgentSpec, ToolRegistry
from engine.tools import run_python, web_fetch, web_search


def section(label: str) -> None:
    pad = (60 - len(label) - 2) // 2
    print(f"\n{'─' * pad} {label} {'─' * pad}\n")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY before running.  See .env.example.")

    # ── Tool registry ─────────────────────────────────────────────────────────
    tools = ToolRegistry()
    tools.register(web_search, web_fetch, run_python)

    @tools.tool
    def word_count(text: str) -> str:
        """Count words in a block of text."""
        return f"{len(text.split())} words"

    # ── Agent specs ───────────────────────────────────────────────────────────
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
        system_prompt=(
            "You receive research notes. Distill them into a 3-bullet executive summary. "
            "Be direct and specific."
        ),
        tools=[],  # no tools needed — just reasoning
    ))

    factory.register(AgentSpec(
        name="code-analyst",
        type="react",
        system_prompt="Use Python to analyse data and produce clear numeric results.",
        tools=["run_python"],
    ))

    factory.register(AgentSpec(
        name="text-analyst",
        type="react",
        system_prompt="Analyse text and report word count, tone, and key themes.",
        tools=["word_count"],
    ))

    # ── Pattern 1: react ─────────────────────────────────────────────────────
    section("PATTERN 1 — react")
    print("Single agent: researcher\n")

    researcher = factory.build("researcher")
    result = researcher.invoke({
        "messages": [{"role": "user", "content": "What is LangGraph used for?"}]
    })
    print(result["messages"][-1].content)

    # ── Pattern 2: sequential ────────────────────────────────────────────────
    section("PATTERN 2 — sequential")
    print("Pipeline: researcher → summariser\n")

    factory.register(AgentSpec(
        name="research-pipeline",
        type="sequential",
        agents=["researcher", "summariser"],
    ))

    pipeline = factory.build("research-pipeline")
    result = pipeline.invoke({
        "messages": [{"role": "user", "content": "What are the main use cases for LangGraph?"}]
    })
    print(result["messages"][-1].content)

    # ── Pattern 3: parallel ──────────────────────────────────────────────────
    section("PATTERN 3 — parallel")
    print("Concurrent: code-analyst + text-analyst on the same input\n")

    factory.register(AgentSpec(
        name="dual-analysis",
        type="parallel",
        agents=["code-analyst", "text-analyst"],
    ))

    parallel_team = factory.build("dual-analysis")
    result = parallel_team.invoke({
        "messages": [{"role": "user", "content": "Analyse this: 'The quick brown fox jumps over the lazy dog.'"}]
    })
    print(result["messages"][-1].content)

    # ── Pattern 4: supervisor (optional) ─────────────────────────────────────
    section("PATTERN 4 — supervisor")
    try:
        factory.register(AgentSpec(
            name="research-team",
            type="supervisor",
            system_prompt=(
                "You manage a research team. "
                "Route web research tasks to 'researcher' and data tasks to 'code-analyst'."
            ),
            agents=["researcher", "code-analyst"],
        ))
        supervisor = factory.build("research-team")
        result = supervisor.invoke({
            "messages": [{"role": "user", "content": "Search for the latest LangGraph release notes."}]
        })
        print(result["messages"][-1].content)
    except ImportError as exc:
        print(f"Skipped: {exc}")

    # ── Pattern 5: swarm (optional) ──────────────────────────────────────────
    section("PATTERN 5 — swarm")
    try:
        factory.register(AgentSpec(
            name="research-swarm",
            type="swarm",
            agents=["researcher", "summariser"],
        ))
        swarm = factory.build("research-swarm")
        result = swarm.invoke({
            "messages": [{"role": "user", "content": "Research and summarise LangGraph's checkpointer feature."}]
        })
        print(result["messages"][-1].content)
    except ImportError as exc:
        print(f"Skipped: {exc}")

    # ── Pattern 6: custom graph via register_graph ────────────────────────────
    section("PATTERN 6 — custom graph")
    print("Manually built StateGraph registered with the factory\n")

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def echo_node(state: State) -> State:
        last = state["messages"][-1].content
        return {"messages": [{"role": "assistant", "content": f"Echo: {last}"}]}

    custom = StateGraph(State)
    custom.add_node("echo", echo_node)
    custom.add_edge(START, "echo")
    custom.add_edge("echo", END)

    factory.register_graph("echo-bot", custom.compile())
    echo = factory.build("echo-bot")
    result = echo.invoke({"messages": [{"role": "user", "content": "hello world"}]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
