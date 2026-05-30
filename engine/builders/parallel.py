from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from ._base import BaseBuilder, MessagesState

if TYPE_CHECKING:
    from ..specs import AgentSpec


class ParallelBuilder(BaseBuilder):
    """Parallelization — all agents run concurrently, outputs merged."""

    def build(self, spec: AgentSpec) -> Any:
        self._require_agents(spec, min_count=2)

        subagents = {name: self._factory.build(name) for name in spec.agents}

        def run_parallel(state):
            def run(item: tuple[str, Any]) -> str:
                name, agent = item
                result = agent.invoke({"messages": state["messages"]})
                return f"[{name}]\n{result['messages'][-1].content}"

            with concurrent.futures.ThreadPoolExecutor() as pool:
                parts = list(pool.map(run, subagents.items()))

            return {"messages": [{"role": "assistant", "content": "\n\n".join(parts)}]}

        graph = StateGraph(MessagesState)
        graph.add_node("run_parallel", run_parallel)
        graph.add_edge(START, "run_parallel")
        graph.add_edge("run_parallel", END)

        return self._compile(graph, spec.name)
