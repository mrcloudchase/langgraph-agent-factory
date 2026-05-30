from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph

from ._base import BaseBuilder, MessagesState

if TYPE_CHECKING:
    from ..specs import AgentSpec


class LlmBuilder(BaseBuilder):
    """Single LLM call — one invocation, returns, done. No tool loop."""

    def build(self, spec: AgentSpec) -> Any:
        llm = self._llm(spec)
        system_prompt = spec.system_prompt

        def node(state):
            messages = list(state["messages"])
            if system_prompt:
                messages = [SystemMessage(content=system_prompt)] + messages
            return {"messages": [llm.invoke(messages)]}

        graph = StateGraph(MessagesState)
        graph.add_node("llm", node)
        graph.add_edge(START, "llm")
        graph.add_edge("llm", END)
        return self._compile(graph, spec.name)
