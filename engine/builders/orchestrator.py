from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._base import BaseBuilder

if TYPE_CHECKING:
    from ..specs import AgentSpec


class OrchestratorBuilder(BaseBuilder):
    """Orchestrator-subagents — LLM supervisor stays in the loop, coordinates dynamically."""

    def build(self, spec: AgentSpec) -> Any:
        try:
            from langgraph_supervisor import create_supervisor
        except ImportError as exc:
            raise ImportError("pip install langgraph-supervisor") from exc

        self._require_steps(spec, min_count=1)

        subagents = [self._factory.build(s) for s in spec.steps]

        graph = create_supervisor(
            agents=subagents,
            model=self._llm(spec),
            prompt=spec.system_prompt or None,
        )
        return self._compile(graph, spec.name)
