from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from ._base import BaseBuilder

if TYPE_CHECKING:
    from ..specs import AgentSpec


class EvaluatorBuilder(BaseBuilder):
    """Evaluator-optimizer — generator produces output, evaluator critiques, loop."""

    def build(self, spec: AgentSpec) -> Any:
        self._require_agents(
            spec, min_count=2, note="agents[0]=generator, agents[1]=evaluator"
        )

        generator = self._factory.build(spec.agents[0])
        evaluator = self._factory.build(spec.agents[1])
        max_iter  = spec.max_iterations

        class State(TypedDict):
            messages:   Annotated[list, add_messages]
            accepted:   bool
            iterations: int

        def generate(state):
            result = generator.invoke({"messages": state["messages"]})
            return {
                "messages":   [result["messages"][-1]],
                "iterations": state.get("iterations", 0) + 1,
                "accepted":   False,
            }

        def evaluate(state):
            result = evaluator.invoke({"messages": state["messages"] + [{
                "role": "user",
                "content": (
                    "Review the latest response above.\n"
                    "Reply ACCEPTED if it meets requirements, or "
                    "REJECTED: <specific feedback> if it needs improvement."
                ),
            }]})
            verdict = result["messages"][-1].content.strip()
            if verdict.upper().startswith("ACCEPTED"):
                return {"accepted": True}
            return {
                "messages": [{"role": "user", "content": f"Feedback: {verdict}. Please revise."}],
                "accepted": False,
            }

        def should_continue(state):
            if state.get("accepted", False) or state.get("iterations", 0) >= max_iter:
                return END
            return "generate"

        graph = StateGraph(State)
        graph.add_node("generate", generate)
        graph.add_node("evaluate", evaluate)
        graph.add_edge(START, "generate")
        graph.add_edge("generate", "evaluate")
        graph.add_conditional_edges("evaluate", should_continue, {END: END, "generate": "generate"})

        return self._compile(graph, spec.name)
