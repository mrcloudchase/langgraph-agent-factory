from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from ._base import BaseBuilder, MessagesState

if TYPE_CHECKING:
    from ..specs import AgentSpec


class ChainBuilder(BaseBuilder):
    """Prompt chaining — A → B → C, output of each step feeds the next."""

    def build(self, spec: AgentSpec) -> Any:
        self._require_agents(spec, min_count=2)

        graph = StateGraph(MessagesState)
        for name in spec.agents:
            graph.add_node(name, self._agent_node(self._factory.build(name)))

        graph.add_edge(START, spec.agents[0])
        for current, next_ in zip(spec.agents, spec.agents[1:]):
            graph.add_edge(current, next_)
        graph.add_edge(spec.agents[-1], END)

        return self._compile(graph, spec.name)
