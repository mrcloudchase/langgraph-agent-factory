# LangGraph Agent Factory

A declarative agent factory that implements every agentic system pattern from Anthropic's [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) using LangGraph.

Define tools, describe agents with `AgentSpec`, and the factory assembles the correct LangGraph graph for you — no boilerplate required.

---

## Patterns

| Type | Anthropic Pattern | How it works |
|------|------------------|--------------|
| `react` | ReAct agent | Single agent: think → call tool → observe → repeat |
| `chain` | Prompt chaining | A → B → C — output of each step feeds the next |
| `router` | Routing | LLM classifies input, dispatches to one specialist, done |
| `parallel` | Parallelization | All agents run concurrently on the same input, outputs merged |
| `orchestrator` | Orchestrator-subagents | LLM stays in the loop, plans and coordinates across agents |
| `evaluator` | Evaluator-optimizer | Generator → critic loop until accepted or max iterations |
| `swarm` | — | Agents hand off to each other freely (LangGraph-specific) |

---

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python main.py          # runs one example of every pattern
```

---

## Usage

### 1. Set up tools

```python
from engine import AgentFactory, AgentSpec, ToolRegistry
from engine.tools import web_search, web_fetch, run_python

tools = ToolRegistry()
tools.register(web_search, web_fetch, run_python)

# Register a custom tool with the .tool decorator
@tools.tool
def word_count(text: str) -> str:
    """Count the words in a text string."""
    return f"{len(text.split())} words"
```

### 2. Define and build agents

```python
factory = AgentFactory(tools)

factory.register(AgentSpec(
    name="researcher",
    type="react",
    system_prompt="Search the web and return concise, cited findings.",
    tools=["web_search", "web_fetch"],
))

agent = factory.build("researcher")
result = agent.invoke({"messages": [{"role": "user", "content": "What is LangGraph?"}]})
print(result["messages"][-1].content)
```

Every agent type is invoked the same way — `agent.invoke({"messages": [...]})`.

---

## Agent types

### `react` — ReAct agent

A single agent that loops: reason about the task, call a tool, observe the result, repeat until done.

```python
factory.register(AgentSpec(
    name="researcher",
    type="react",
    system_prompt="Search the web and return concise, cited findings.",
    tools=["web_search", "web_fetch"],
))
```

---

### `chain` — Prompt chaining

Agents run in sequence. The last message from each step becomes the input to the next.

```
researcher → summariser → formatter
```

```python
factory.register(AgentSpec(
    name="research-pipeline",
    type="chain",
    agents=["researcher", "summariser"],
))
```

---

### `router` — Routing

An LLM classifier reads the input and routes it to exactly one specialist agent. The classifier makes one decision — it does not stay in the loop.

```
input → [classify] → researcher
                   ↘ coder
                   ↘ writer
```

```python
factory.register(AgentSpec(
    name="smart-router",
    type="router",
    agents=["researcher", "coder", "writer"],
))
```

The classifier uses each agent's `system_prompt` as its description when deciding which to pick.

---

### `parallel` — Parallelization

All agents receive the same input and run concurrently. Their outputs are collected and merged into a single response.

```
         ┌→ analyst-a ─┐
input ───┤              ├→ merged output
         └→ analyst-b ─┘
```

```python
factory.register(AgentSpec(
    name="dual-analysis",
    type="parallel",
    agents=["analyst-a", "analyst-b"],
))
```

---

### `orchestrator` — Orchestrator-subagents

An orchestrator LLM dynamically decides which agent to call at each step and synthesises the final result. Unlike routing, it stays in control across multiple turns.

Requires: `pip install langgraph-supervisor`

```python
factory.register(AgentSpec(
    name="research-team",
    type="orchestrator",
    system_prompt=(
        "Coordinate the team. Use researcher for web lookups, "
        "coder for calculations, writer for the final answer."
    ),
    agents=["researcher", "coder", "writer"],
))
```

---

### `evaluator` — Evaluator-optimizer

`agents[0]` generates output. `agents[1]` reviews it and replies `ACCEPTED` or `REJECTED: <feedback>`. On rejection the feedback is added to the conversation and the generator tries again. Exits when accepted or `max_iterations` is reached.

```
START → generate → evaluate → ACCEPTED? → END
            ↑                ↘ feedback ↗
```

```python
factory.register(AgentSpec(
    name="write-and-review",
    type="evaluator",
    agents=["writer", "critic"],   # [0] generator, [1] evaluator
    max_iterations=3,
))
```

The evaluator agent's `system_prompt` should instruct it to reply with `ACCEPTED` or `REJECTED: <specific feedback>`.

---

### `swarm` — Swarm

Each agent in the pool can hand off to any other agent. The factory automatically creates handoff tools between all participants. Control starts with `agents[0]`.

Requires: `pip install langgraph-swarm`

```python
factory.register(AgentSpec(
    name="research-swarm",
    type="swarm",
    agents=["researcher", "summariser"],
))
```

---

## Custom workflows

For patterns not covered above, build a LangGraph `StateGraph` directly and register it with the factory by name:

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

def my_node(state):
    return {"messages": [{"role": "assistant", "content": "done"}]}

graph = StateGraph(State)
graph.add_node("step", my_node)
graph.add_edge(START, "step")
graph.add_edge("step", END)

factory.register_graph("my-workflow", graph.compile())
agent = factory.build("my-workflow")
```

---

## Built-in tools

Three general-purpose tools ship with the factory and cover most use cases without domain-specific integrations:

| Tool | Description |
|------|-------------|
| `web_search` | Search the web via DuckDuckGo, returns titles, URLs, and snippets |
| `web_fetch` | Fetch a URL and extract its readable text (strips nav, scripts, boilerplate) |
| `run_python` | Execute Python code and return its stdout output |

```python
from engine.tools import web_search, web_fetch, run_python

tools = ToolRegistry()
tools.register(web_search, web_fetch, run_python)
```

---

## `AgentSpec` reference

| Field | Type | Default | Used by |
|-------|------|---------|---------|
| `name` | `str` | required | all |
| `type` | `str` | `"react"` | all |
| `model` | `str` | `"claude-opus-4-8"` | all |
| `system_prompt` | `str` | `""` | all |
| `tools` | `list[str]` | `[]` | `react`, `swarm` |
| `agents` | `list[str]` | `[]` | `chain`, `router`, `parallel`, `orchestrator`, `evaluator`, `swarm` |
| `max_iterations` | `int` | `5` | `evaluator` |
| `checkpointer` | `bool` | `False` | `react` (enables memory) |

---

## Requirements

```
ANTHROPIC_API_KEY   required for all patterns
langgraph-supervisor   required for orchestrator
langgraph-swarm        required for swarm
```
