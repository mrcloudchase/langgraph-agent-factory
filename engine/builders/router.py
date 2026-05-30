from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from ._base import BaseBuilder

if TYPE_CHECKING:
    from ..specs import AgentSpec


class RouterBuilder(BaseBuilder):
    """Routing — LLM classifies the input, code dispatches to one step, done."""

    def build(self, spec: AgentSpec) -> Any:
        self._require_steps(spec, min_count=2)

        step_names = [s.name for s in spec.steps]
        step_agents = {s.name: self._factory.build(s) for s in spec.steps}

        options = "\n".join(
            f"  {s.name}: {(s.system_prompt.splitlines()[0] if s.system_prompt else s.name)}"
            for s in spec.steps
        )
        llm = self._llm(spec)

        class State(TypedDict):
            messages: Annotated[list, add_messages]
            route: str

        def classify(state):
            response = llm.invoke(state["messages"] + [{
                "role": "user",
                "content": (
                    f"Choose the best agent for this request.\n\n"
                    f"Agents:\n{options}\n\n"
                    f"Reply with only the agent name — one of: {', '.join(step_names)}"
                ),
            }])
            chosen = response.content.strip()
            if chosen in step_names:
                return {"route": chosen}
            for name in step_names:
                if name.lower() in chosen.lower():
                    return {"route": name}
            return {"route": step_names[0]}

        graph = StateGraph(State)
        graph.add_node("classify", classify)
        graph.add_edge(START, "classify")

        for name, agent in step_agents.items():
            graph.add_node(name, self._agent_node(agent))
            graph.add_edge(name, END)

        graph.add_conditional_edges("classify", lambda s: s["route"],
                                    {name: name for name in step_names})

        return self._compile(graph, spec.name)
