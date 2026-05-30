from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from ._base import BaseBuilder, MessagesState

if TYPE_CHECKING:
    from ..specs import AgentSpec


class ChainBuilder(BaseBuilder):
    """Prompt chaining — steps run in sequence, each output fed to the next."""

    def build(self, spec: AgentSpec) -> Any:
        self._require_steps(spec, min_count=2)

        graph = StateGraph(MessagesState)
        for step in spec.steps:
            graph.add_node(step.name, self._agent_node(self._factory.build(step)))

        graph.add_edge(START, spec.steps[0].name)
        for cur, nxt in zip(spec.steps, spec.steps[1:]):
            graph.add_edge(cur.name, nxt.name)
        graph.add_edge(spec.steps[-1].name, END)

        return self._compile(graph, spec.name)
