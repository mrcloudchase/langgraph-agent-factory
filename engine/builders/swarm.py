from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from ._base import BaseBuilder

if TYPE_CHECKING:
    from ..specs import AgentSpec


class SwarmBuilder(BaseBuilder):
    """Swarm — agents hand off to each other freely; any agent can start."""

    def build(self, spec: AgentSpec) -> Any:
        try:
            from langgraph_swarm import create_swarm, create_handoff_tool
        except ImportError as exc:
            raise ImportError("pip install langgraph-swarm") from exc

        self._require_agents(spec)

        subagents = []
        for name in spec.agents:
            subspec   = self._factory.get_spec(name)
            own_tools = [self._factory.get_tool(t) for t in subspec.tools]
            handoffs  = [
                create_handoff_tool(agent_name=other)
                for other in spec.agents if other != name
            ]
            agent = create_react_agent(
                model=ChatAnthropic(model=subspec.model),
                tools=own_tools + handoffs,
                name=name,
                prompt=subspec.system_prompt or None,
            )
            subagents.append(agent)

        compiled = create_swarm(
            agents=subagents,
            default_active_agent=spec.agents[0],
        ).compile()
        compiled.name = spec.name
        return compiled
