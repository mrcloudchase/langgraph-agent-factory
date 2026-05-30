from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from ..factory import AgentFactory
    from ..specs import AgentSpec


class MessagesState(TypedDict):
    """Shared state for workflow patterns that only need a message list."""
    messages: Annotated[list, add_messages]


class BaseBuilder:
    """Common interface and shared helpers for every agent builder."""

    def __init__(self, factory: AgentFactory) -> None:
        self._factory = factory

    def build(self, spec: AgentSpec) -> Any:
        raise NotImplementedError

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _llm(self, spec: AgentSpec) -> ChatAnthropic:
        return ChatAnthropic(model=spec.model)

    def _subagents(self, spec: AgentSpec) -> list[Any]:
        return [self._factory.build(name) for name in spec.agents]

    def _compile(self, graph: StateGraph, name: str) -> Any:
        compiled = graph.compile()
        compiled.name = name
        return compiled

    def _require_agents(self, spec: AgentSpec, min_count: int = 1, note: str = "") -> None:
        if len(spec.agents) < min_count:
            suffix = f" ({note})" if note else ""
            raise ValueError(
                f"'{spec.name}' ({spec.type}) requires ≥{min_count} agent(s){suffix}. "
                f"Got: {spec.agents}"
            )
        for name in spec.agents:
            if not self._factory.has_spec(name):
                raise KeyError(
                    f"Subagent '{name}' referenced by '{spec.name}' is not registered."
                )

    @staticmethod
    def _agent_node(agent: Any):
        """Wrap a compiled agent as a StateGraph node.

        Invokes the agent on the current message list and appends its last
        reply — the standard handoff between steps in chain and router.
        """
        def node(state):
            result = agent.invoke({"messages": state["messages"]})
            return {"messages": [result["messages"][-1]]}
        return node
