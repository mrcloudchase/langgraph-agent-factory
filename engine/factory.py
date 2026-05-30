"""AgentFactory — builds any LangGraph agentic system from an AgentSpec.

Implements all five workflow patterns from Anthropic's "Building Effective Agents"
plus the ReAct agent and the LangGraph swarm pattern:

  react         Think → act (tool call) → observe → repeat
  chain         Prompt chaining: A → B → C, output fed forward each step
  router        Classify input once, dispatch to one specialised agent
  parallel      All agents run concurrently on the same input, outputs merged
  orchestrator  Orchestrator LLM dynamically plans and directs subagents
  evaluator     Generator → evaluator → loop until accepted or max iterations
  swarm         Agents hand off to each other freely (LangGraph-specific)

Custom workflows
----------------
Build any StateGraph yourself and hand it to the factory:

    graph = StateGraph(State)
    ...
    factory.register_graph("my-workflow", graph.compile())
    result = factory.build("my-workflow").invoke(...)
"""

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
    def __init__(self, tools: ToolRegistry) -> None:
        self._tools = tools
        self._specs: dict[str, AgentSpec] = {}
        self._graphs: dict[str, Any] = {}

    def register(self, spec: AgentSpec) -> None:
        """Register an AgentSpec so it can be built by name later."""
        self._specs[spec.name] = spec

    def register_graph(self, name: str, graph: Any) -> None:
        """Register a pre-built compiled graph (custom StateGraph workflows)."""
        self._graphs[name] = graph

    def build(self, spec: AgentSpec | str) -> Any:
        """Build and return a compiled, invokable agent graph."""
        if isinstance(spec, str):
            if spec in self._graphs:
                return self._graphs[spec]
            if spec not in self._specs:
                raise KeyError(
                    f"Agent '{spec}' not registered. Available: {sorted(self._specs)}"
                )
            spec = self._specs[spec]

        builders = {
            "react":        self._build_react,
            "chain":        self._build_chain,
            "sequential":   self._build_chain,        # alias
            "router":       self._build_router,
            "parallel":     self._build_parallel,
            "orchestrator": self._build_orchestrator,
            "supervisor":   self._build_orchestrator, # alias
            "evaluator":    self._build_evaluator,
            "swarm":        self._build_swarm,
        }
        return builders[spec.type](spec)

    # ── Pattern 1: ReAct ──────────────────────────────────────────────────────

    def _build_react(self, spec: AgentSpec) -> Any:
        """Single agent that loops: think → call tool → observe → repeat."""
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

    # ── Pattern 2: Prompt chaining ────────────────────────────────────────────

    def _build_chain(self, spec: AgentSpec) -> Any:
        """Output of each step becomes the input to the next.

        START → agent_0 → agent_1 → … → agent_n → END
        """
        self._require_agents(spec, min_count=2)

        class State(TypedDict):
            messages: Annotated[list, add_messages]

        graph = StateGraph(State)

        for name in spec.agents:
            agent = self._get_built(name)

            def make_node(ag: Any):
                def node(state) -> dict:
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

    # ── Pattern 3: Routing ────────────────────────────────────────────────────

    def _build_router(self, spec: AgentSpec) -> Any:
        """Classify the input once, route to exactly one specialised agent.

        START → router → (one of) agent_0 | agent_1 | … → END
        """
        self._require_agents(spec, min_count=2)

        class State(TypedDict):
            messages: Annotated[list, add_messages]
            route: str

        agent_names = spec.agents
        llm = ChatAnthropic(model=spec.model)

        options = "\n".join(
            f"  {n}: {self._specs[n].system_prompt[:120] or n}"
            for n in agent_names
        )

        def router_node(state) -> dict:
            msg = {
                "role": "user",
                "content": (
                    f"Pick the single best agent for this request.\n\n"
                    f"Agents:\n{options}\n\n"
                    "Reply with only the agent name — nothing else."
                ),
            }
            response = llm.invoke(state["messages"] + [msg])
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
            agent = self._get_built(name)

            def make_node(ag: Any):
                def node(state) -> dict:
                    result = ag.invoke({"messages": state["messages"]})
                    return {"messages": [result["messages"][-1]]}
                return node

            graph.add_node(name, make_node(agent))
            graph.add_edge(name, END)

        graph.add_conditional_edges(
            "router",
            lambda state: state["route"],
            {name: name for name in agent_names},
        )

        compiled = graph.compile()
        compiled.name = spec.name
        return compiled

    # ── Pattern 4: Parallelization ────────────────────────────────────────────

    def _build_parallel(self, spec: AgentSpec) -> Any:
        """All agents run concurrently on the same input; outputs are merged.

        START → [agent_0 ‖ agent_1 ‖ …] → merged output → END
        """
        self._require_agents(spec, min_count=2)

        subagents = {name: self._get_built(name) for name in spec.agents}

        class State(TypedDict):
            messages: Annotated[list, add_messages]

        def run_parallel(state) -> dict:
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

    # ── Pattern 5: Orchestrator-subagents ────────────────────────────────────

    def _build_orchestrator(self, spec: AgentSpec) -> Any:
        """Orchestrator LLM dynamically plans, dispatches, and synthesises results.

        The orchestrator decides at each step which agent to call next and
        when the task is complete — unlike routing, it stays in the loop.
        """
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

    # ── Pattern 6: Evaluator-optimizer ───────────────────────────────────────

    def _build_evaluator(self, spec: AgentSpec) -> Any:
        """Generator produces output; evaluator critiques; loop until accepted.

        agents[0] = generator
        agents[1] = evaluator

        START → generate → evaluate → ACCEPTED? → END
                    ↑                ↘ feedback ↗
        """
        self._require_agents(spec, min_count=2, label="agents[0]=generator, agents[1]=evaluator")

        generator = self._get_built(spec.agents[0])
        evaluator = self._get_built(spec.agents[1])
        max_iter  = spec.max_iterations

        class State(TypedDict):
            messages:   Annotated[list, add_messages]
            accepted:   bool
            iterations: int

        def generate(state) -> dict:
            result = generator.invoke({"messages": state["messages"]})
            return {
                "messages":   [result["messages"][-1]],
                "iterations": state.get("iterations", 0) + 1,
                "accepted":   False,
            }

        def evaluate(state) -> dict:
            eval_input = state["messages"] + [{
                "role": "user",
                "content": (
                    "Review the latest response above.\n"
                    "Reply ACCEPTED if it meets requirements, or "
                    "REJECTED: <specific feedback> if it needs improvement."
                ),
            }]
            result  = evaluator.invoke({"messages": eval_input})
            verdict = result["messages"][-1].content.strip()
            if verdict.upper().startswith("ACCEPTED"):
                return {"accepted": True}
            return {
                "messages": [{"role": "user", "content": f"Feedback: {verdict}. Please revise."}],
                "accepted": False,
            }

        def should_continue(state) -> str:
            if state.get("accepted", False) or state.get("iterations", 0) >= max_iter:
                return END
            return "generate"

        graph = StateGraph(State)
        graph.add_node("generate", generate)
        graph.add_node("evaluate", evaluate)
        graph.add_edge(START, "generate")
        graph.add_edge("generate", "evaluate")
        graph.add_conditional_edges("evaluate", should_continue, {END: END, "generate": "generate"})

        compiled = graph.compile()
        compiled.name = spec.name
        return compiled

    # ── LangGraph bonus: Swarm ────────────────────────────────────────────────

    def _build_swarm(self, spec: AgentSpec) -> Any:
        """Agents hand off to each other; any agent can start or receive control.

        Each agent gets handoff tools to all other agents in the swarm.
        """
        try:
            from langgraph_swarm import create_swarm, create_handoff_tool
        except ImportError as exc:
            raise ImportError("pip install langgraph-swarm") from exc

        self._require_agents(spec)

        subagents = []
        for name in spec.agents:
            subspec   = self._specs[name]
            own_tools = [self._tools.get(t) for t in subspec.tools]
            handoffs  = [
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

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _require_agents(self, spec: AgentSpec, min_count: int = 1, label: str = "") -> None:
        if len(spec.agents) < min_count:
            suffix = f" ({label})" if label else ""
            raise ValueError(
                f"'{spec.name}' ({spec.type}) needs at least {min_count} agent(s){suffix}. "
                f"Got: {spec.agents}"
            )
        for name in spec.agents:
            if name not in self._specs:
                raise KeyError(
                    f"Subagent '{name}' referenced by '{spec.name}' is not registered."
                )

    def _get_built(self, name: str) -> Any:
        return self.build(self._specs[name])
