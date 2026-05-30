from __future__ import annotations

from typing import Any

from .builders import BUILDERS
from .registry import ToolRegistry
from .specs import AgentSpec


class AgentFactory:
    """Builds any LangGraph agentic system from an AgentSpec.

        factory = AgentFactory(tools)
        factory.register(spec)
        agent = factory.build("my-agent")
        result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
    """

    def __init__(self, tools: ToolRegistry) -> None:
        self._tools = tools
        self._specs: dict[str, AgentSpec] = {}
        self._graphs: dict[str, Any] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, spec: AgentSpec) -> None:
        """Register an AgentSpec so it can be built by name."""
        self._specs[spec.name] = spec

    def register_graph(self, name: str, graph: Any) -> None:
        """Register a pre-built compiled graph (custom StateGraph workflows)."""
        self._graphs[name] = graph

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, spec: AgentSpec | str) -> Any:
        """Build and return a compiled, invokable agent graph."""
        if isinstance(spec, str):
            if spec in self._graphs:
                return self._graphs[spec]
            if spec not in self._specs:
                raise KeyError(
                    f"Agent '{spec}' not registered. Available: {sorted(self._specs)}"
                )
            spec = self._specs[spec]
        return BUILDERS[spec.type](self).build(spec)

    # ── Accessors for builders ────────────────────────────────────────────────

    def has_spec(self, name: str) -> bool:
        return name in self._specs

    def get_spec(self, name: str) -> AgentSpec:
        return self._specs[name]

    def get_tool(self, name: str) -> Any:
        return self._tools.get(name)
