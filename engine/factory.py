from __future__ import annotations

import concurrent.futures
from typing import Annotated, Any

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

from .registry import ToolRegistry
from .specs import AgentSpec


class AgentFactory:
    """Builds runnable LangGraph agents from AgentSpecs.

    Supported types
    ---------------
    react       Single agent with tools (create_react_agent).
    supervisor  One routing agent delegates to specialised subagents.
    swarm       Agents hand off to each other; any can start.
    sequential  Agents run in order; each receives the previous output.
    parallel    All agents run concurrently; outputs are merged.

    Custom workflows
    ----------------
    Build any StateGraph yourself and register it with register_graph():

        graph = StateGraph(State)
        ...
        factory.register_graph("my-workflow", graph.compile())
        agent = factory.build("my-workflow")
    """

    def __init__(self, tools: ToolRegistry) -> None:
        self._tools = tools
        self._specs: dict[str, AgentSpec] = {}
        self._graphs: dict[str, Any] = {}

    def register(self, spec: AgentSpec) -> None:
        """Register an AgentSpec so it can be built by name."""
        self._specs[spec.name] = spec

    def register_graph(self, name: str, graph: Any) -> None:
        """Register a pre-built compiled graph for custom workflows."""
        self._graphs[name] = graph

    def build(self, spec: AgentSpec | str):
        """Build and return a compiled agent graph."""
        if isinstance(spec, str):
            if spec in self._graphs:
                return self._graphs[spec]
            if spec not in self._specs:
                raise KeyError(
                    f"Agent '{spec}' not registered. "
                    f"Available: {sorted(self._specs)}"
                )
            spec = self._specs[spec]

        builders = {
            "react":      self._build_react,
            "supervisor": self._build_supervisor,
            "swarm":      self._build_swarm,
            "sequential": self._build_sequential,
            "parallel":   self._build_parallel,
        }
        return builders[spec.type](spec)

    # ── Builders ──────────────────────────────────────────────────────────────

    def _build_react(self, spec: AgentSpec):
        tools = [self._tools.get(t) for t in spec.tools]
        kwargs: dict = {
            "model": ChatAnthropic(model=spec.model),
            "tools": tools,
            "name":  spec.name,
        }
        if spec.system_prompt:
            kwargs["prompt"] = spec.system_prompt
        if spec.checkpointer:
            kwargs["checkpointer"] = MemorySaver()
        return create_react_agent(**kwargs)

    def _build_supervisor(self, spec: AgentSpec):
        try:
            from langgraph_supervisor import create_supervisor
        except ImportError as exc:
            raise ImportError("pip install langgraph-supervisor") from exc

        self._require_agents(spec)
        subagents = [self._get_built(name) for name in spec.agents]

        graph = create_supervisor(
            agents=subagents,
            model=ChatAnthropic(model=spec.model),
            prompt=spec.system_prompt or None,
        )
        compiled = graph.compile()
        compiled.name = spec.name
        return compiled

    def _build_swarm(self, spec: AgentSpec):
        try:
            from langgraph_swarm import create_swarm, create_handoff_tool
        except ImportError as exc:
            raise ImportError("pip install langgraph-swarm") from exc

        self._require_agents(spec)

        subagents = []
        for name in spec.agents:
            subspec = self._specs[name]
            own_tools = [self._tools.get(t) for t in subspec.tools]
            handoffs = [
                create_handoff_tool(agent_name=other)
                for other in spec.agents
                if other != name
            ]
            agent = create_react_agent(
                model=ChatAnthropic(model=subspec.model),
                tools=own_tools + handoffs,
                name=name,
                prompt=subspec.system_prompt or None,
            )
            subagents.append(agent)

        compiled = create_swarm(
            agents=subagents,
            default_active_agent=spec.agents[0],
        ).compile()
        compiled.name = spec.name
        return compiled

    def _build_sequential(self, spec: AgentSpec):
        """Chain agents so each receives the previous agent's last message."""
        self._require_agents(spec)

        class State(TypedDict):
            messages: Annotated[list, add_messages]

        graph = StateGraph(State)

        for name in spec.agents:
            agent = self._get_built(name)

            def make_node(ag):
                def node(state: State):
                    result = ag.invoke({"messages": state["messages"]})
                    return {"messages": [result["messages"][-1]]}
                return node

            graph.add_node(name, make_node(agent))

        graph.add_edge(START, spec.agents[0])
        for i in range(len(spec.agents) - 1):
            graph.add_edge(spec.agents[i], spec.agents[i + 1])
        graph.add_edge(spec.agents[-1], END)

        compiled = graph.compile()
        compiled.name = spec.name
        return compiled

    def _build_parallel(self, spec: AgentSpec):
        """Run all agents concurrently on the same input; merge their outputs."""
        self._require_agents(spec)

        subagents = {name: self._get_built(name) for name in spec.agents}

        class State(TypedDict):
            messages: Annotated[list, add_messages]

        def run_parallel(state: State) -> State:
            def run(item: tuple[str, Any]) -> str:
                name, agent = item
                result = agent.invoke({"messages": state["messages"]})
                return f"[{name}]\n{result['messages'][-1].content}"

            with concurrent.futures.ThreadPoolExecutor() as pool:
                parts = list(pool.map(run, subagents.items()))

            return {"messages": [{"role": "assistant", "content": "\n\n".join(parts)}]}

        graph = StateGraph(State)
        graph.add_node("run_parallel", run_parallel)
        graph.add_edge(START, "run_parallel")
        graph.add_edge("run_parallel", END)

        compiled = graph.compile()
        compiled.name = spec.name
        return compiled

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _require_agents(self, spec: AgentSpec) -> None:
        if not spec.agents:
            raise ValueError(f"'{spec.name}' ({spec.type}) must list at least one agent.")
        for name in spec.agents:
            if name not in self._specs:
                raise KeyError(
                    f"Subagent '{name}' referenced by '{spec.name}' is not registered."
                )

    def _get_built(self, name: str):
        return self.build(self._specs[name])
