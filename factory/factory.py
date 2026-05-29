"""AgentFactory — turns a declarative YAML spec into a compiled LangGraph app.

Supported spec types (MVP):
  - type: agent              -> a ReAct agent (LLM decides what to do)
  - type: workflow           -> a graph of agents wired in a fixed order
      structure: sequential  -> node_0 -> node_1 -> ... -> node_n

Every build returns a compiled graph with the same `.invoke()` / `.stream()`
interface, so callers never need to know which kind they got.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent

from .tools import TOOL_REGISTRY


class AgentFactory:
    def __init__(self, tool_registry: dict | None = None, default_model: str = "claude-opus-4-8"):
        self.tools = tool_registry or TOOL_REGISTRY
        self.default_model = default_model

    # ---- public API ----------------------------------------------------
    def from_yaml(self, path: str | Path):
        """Load a YAML spec from disk and build the corresponding graph."""
        spec = yaml.safe_load(Path(path).read_text())
        return self.build(spec)

    def build(self, spec: dict):
        kind = spec.get("type")
        if kind == "agent":
            return self._build_agent(spec)
        if kind == "workflow":
            return self._build_workflow(spec)
        raise ValueError(f"Unknown type {kind!r} (expected 'agent' or 'workflow')")

    # ---- builders ------------------------------------------------------
    def _build_agent(self, spec: dict):
        return create_react_agent(
            model=self._model(spec.get("model")),
            tools=self._resolve_tools(spec.get("tools")),
            prompt=spec.get("prompt"),
            name=spec["name"],
            checkpointer=self._checkpointer(spec.get("memory")),
        )

    def _build_workflow(self, spec: dict):
        structure = spec.get("structure", "sequential")
        if structure != "sequential":
            raise NotImplementedError(f"MVP supports structure 'sequential' only, got {structure!r}")

        nodes = spec["nodes"]
        builder = StateGraph(MessagesState)

        # Each node is itself a small ReAct agent with its own prompt + tools.
        for node in nodes:
            sub_agent = create_react_agent(
                model=self._model(spec.get("model")),
                tools=self._resolve_tools(node.get("tools")),
                prompt=node["prompt"],
                name=node["name"],
            )
            builder.add_node(node["name"], self._as_node(sub_agent))

        # Wire START -> n0 -> n1 -> ... -> nN -> END
        builder.add_edge(START, nodes[0]["name"])
        for prev, nxt in zip(nodes, nodes[1:]):
            builder.add_edge(prev["name"], nxt["name"])
        builder.add_edge(nodes[-1]["name"], END)

        # Any node flagged interrupt_before pauses the graph for human approval.
        interrupts = [n["name"] for n in nodes if n.get("interrupt_before")]
        return builder.compile(
            checkpointer=self._checkpointer(spec.get("memory")),
            interrupt_before=interrupts or None,
        )

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _as_node(sub_agent):
        """Adapt a compiled sub-agent so it behaves as one node in the outer graph."""

        def node(state: MessagesState):
            result = sub_agent.invoke({"messages": state["messages"]})
            # Return only the messages this node added; the reducer appends them.
            new_messages = result["messages"][len(state["messages"]):]
            return {"messages": new_messages}

        return node

    def _model(self, name: str | None):
        return init_chat_model(name or self.default_model, model_provider="anthropic")

    def _resolve_tools(self, names):
        return [self.tools[n] for n in (names or [])]

    @staticmethod
    def _checkpointer(memory: dict | None):
        if not memory:
            return None
        if memory.get("type") == "memory":
            return MemorySaver()
        # Real deployments would add 'sqlite' / 'postgres' here.
        raise NotImplementedError("MVP supports memory.type 'memory' only")
