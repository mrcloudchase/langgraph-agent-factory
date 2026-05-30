from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from ._base import BaseBuilder

if TYPE_CHECKING:
    from ..specs import AgentSpec


class RouterBuilder(BaseBuilder):
    """Routing — LLM classifies once, dispatches to one specialist, done."""

    def build(self, spec: AgentSpec) -> Any:
        self._require_agents(spec, min_count=2)

        class State(TypedDict):
            messages: Annotated[list, add_messages]
            route: str

        agent_names = spec.agents
        llm = self._llm(spec)
        options = "\n".join(
            f"  {n}: {(self._factory.get_spec(n).system_prompt[:120] or n).rstrip()}"
            for n in agent_names
        )

        def router_node(state):
            response = llm.invoke(state["messages"] + [{
                "role": "user",
                "content": (
                    f"Pick the single best agent for this request.\n\n"
                    f"Agents:\n{options}\n\n"
                    "Reply with only the agent name — nothing else."
                ),
            }])
            chosen = response.content.strip()
            if chosen in agent_names:
                return {"route": chosen}
            for name in agent_names:
                if name.lower() in chosen.lower():
                    return {"route": name}
            return {"route": agent_names[0]}

        graph = StateGraph(State)
        graph.add_node("router", router_node)
        graph.add_edge(START, "router")

        for name in agent_names:
            graph.add_node(name, self._agent_node(self._factory.build(name)))
            graph.add_edge(name, END)

        graph.add_conditional_edges(
            "router",
            lambda s: s["route"],
            {name: name for name in agent_names},
        )

        return self._compile(graph, spec.name)
