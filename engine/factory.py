from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .builders import BUILDERS
from .registry import ToolRegistry
from .specs import AgentSpec


class AgentFactory:
    """Builds any LangGraph agentic system from YAML agent definitions.

    Typical usage:

        factory = AgentFactory(tools)
        factory.load("agents/")           # load all *.yaml files in a directory
        agent = factory.build("researcher")
        result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})

    You can also register specs programmatically:

        factory.register(AgentSpec(name="my-agent", type="react", ...))
    """

    def __init__(self, tools: ToolRegistry) -> None:
        self._tools = tools
        self._specs: dict[str, AgentSpec] = {}
        self._graphs: dict[str, Any] = {}

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self, path: str | Path) -> None:
        """Load agent specs from a YAML file or every *.yaml file in a directory."""
        p = Path(path)
        if p.is_dir():
            for f in sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml")):  # type: ignore[operator]
                self._load_file(f)
        elif p.is_file():
            self._load_file(p)
        else:
            raise FileNotFoundError(f"Path not found: {path}")

    def _load_file(self, path: Path) -> None:
        try:
            data = yaml.safe_load(path.read_text())
            self.register(AgentSpec(**data))
        except ValidationError as exc:
            raise ValueError(f"Invalid agent spec in '{path.name}':\n{exc}") from exc
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in '{path.name}': {exc}") from exc

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, spec: AgentSpec) -> None:
        """Register an AgentSpec programmatically."""
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

