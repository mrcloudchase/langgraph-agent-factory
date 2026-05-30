from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._base import BaseBuilder

if TYPE_CHECKING:
    from ..specs import AgentSpec


class OrchestratorBuilder(BaseBuilder):
    """Orchestrator-subagents — LLM stays in the loop, plans and coordinates."""

    def build(self, spec: AgentSpec) -> Any:
        try:
            from langgraph_supervisor import create_supervisor
        except ImportError as exc:
            raise ImportError("pip install langgraph-supervisor") from exc

        self._require_agents(spec)

        graph = create_supervisor(
            agents=self._subagents(spec),
            model=self._llm(spec),
            prompt=spec.system_prompt or None,
        )
        return self._compile(graph, spec.name)
