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
    messages: Annotated[list, add_messages]


class BaseBuilder:
    def __init__(self, factory: AgentFactory) -> None:
        self._factory = factory

    def build(self, spec: AgentSpec) -> Any:
        raise NotImplementedError

    def _llm(self, spec: AgentSpec) -> ChatAnthropic:
        return ChatAnthropic(model=spec.model)

    def _compile(self, graph: StateGraph, name: str) -> Any:
        compiled = graph.compile()
        compiled.name = name
        return compiled

    def _require_steps(self, spec: AgentSpec, min_count: int = 1) -> None:
        if len(spec.steps) < min_count:
            raise ValueError(
                f"'{spec.name}' (type={spec.type}) requires at least "
                f"{min_count} step(s), got {len(spec.steps)}."
            )

    @staticmethod
    def _agent_node(agent: Any):
        def node(state):
            result = agent.invoke({"messages": state["messages"]})
            return {"messages": [result["messages"][-1]]}
        return node
