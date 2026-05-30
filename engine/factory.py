from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from .registry import ToolRegistry
from .specs import AgentSpec


class AgentFactory:
    """Builds runnable LangGraph agents from AgentSpecs.

    Usage:
        factory = AgentFactory(tools)
        factory.register(spec)
        agent = factory.build("my-agent")   # by name
        agent = factory.build(spec)         # or directly from spec
        result = agent.invoke({"messages": [...]})
    """

    def __init__(self, tools: ToolRegistry) -> None:
        self._tools = tools
        self._specs: dict[str, AgentSpec] = {}

    def register(self, spec: AgentSpec) -> None:
        """Add a spec to the factory so it can be built by name."""
        self._specs[spec.name] = spec

    def build(self, spec: AgentSpec | str):
        """Build and return a compiled agent graph."""
        if isinstance(spec, str):
            if spec not in self._specs:
                raise KeyError(
                    f"Agent '{spec}' not registered. "
                    f"Available: {sorted(self._specs)}"
                )
            spec = self._specs[spec]

        if spec.type == "react":
            return self._build_react(spec)
        if spec.type == "supervisor":
            return self._build_supervisor(spec)
        raise ValueError(f"Unknown agent type: {spec.type!r}")

    def _build_react(self, spec: AgentSpec):
        tools = [self._tools.get(name) for name in spec.tools]
        kwargs: dict = {
            "model": ChatAnthropic(model=spec.model),
            "tools": tools,
        }
        if spec.system_prompt:
            kwargs["prompt"] = spec.system_prompt
        if spec.checkpointer:
            kwargs["checkpointer"] = MemorySaver()
        agent = create_react_agent(**kwargs)
        agent.name = spec.name
        return agent

    def _build_supervisor(self, spec: AgentSpec):
        try:
            from langgraph_supervisor import create_supervisor
        except ImportError as exc:
            raise ImportError(
                "Supervisor agents require: pip install langgraph-supervisor"
            ) from exc

        if not spec.agents:
            raise ValueError(
                f"Supervisor '{spec.name}' must list at least one agent in 'agents'."
            )

        subagents = []
        for name in spec.agents:
            if name not in self._specs:
                raise KeyError(
                    f"Subagent '{name}' referenced by supervisor '{spec.name}' "
                    f"is not registered."
                )
            subagents.append(self.build(self._specs[name]))

        graph = create_supervisor(
            agents=subagents,
            model=ChatAnthropic(model=spec.model),
            prompt=spec.system_prompt or None,
        )
        compiled = graph.compile()
        compiled.name = spec.name
        return compiled
