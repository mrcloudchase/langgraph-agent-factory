from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from ._base import BaseBuilder

if TYPE_CHECKING:
    from ..specs import AgentSpec


class ReactBuilder(BaseBuilder):
    """ReAct agent — think → call tool → observe → repeat."""

    def build(self, spec: AgentSpec) -> Any:
        tools = [self._factory.get_tool(t) for t in spec.tools]
        kwargs: dict = {
            "model": self._llm(spec),
            "tools": tools,
            "name":  spec.name,
        }
        if spec.system_prompt:
            kwargs["prompt"] = spec.system_prompt
        if spec.checkpointer:
            kwargs["checkpointer"] = MemorySaver()
        return create_react_agent(**kwargs)
