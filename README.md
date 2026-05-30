# LangGraph Agent Factory

A YAML-driven agent factory that implements every agentic system pattern from Anthropic's [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) using LangGraph.

Define agents in YAML. Register tools in Python. The factory handles the rest.

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
python main.py
```

---

## Usage

### 1. Register tools

Tools are Python functions — they have to be, since they execute code. Register built-ins or write your own:

```python
from engine import AgentFactory, ToolRegistry
from engine.tools import web_search, web_fetch, run_python

tools = ToolRegistry()
tools.register(web_search, web_fetch, run_python)

@tools.tool
def word_count(text: str) -> str:
    """Count the words in a text string."""
    return f"{len(text.split())} words"
```

### 2. Define agents in YAML

Create a file in `agents/` for each agent. The filename is up to you — the `name` field is what the factory uses.

```yaml
# agents/researcher.yaml
name: researcher
type: react
system_prompt: Search the web and return concise, cited findings.
tools:
  - web_search
  - web_fetch
```

```yaml
# agents/research-chain.yaml
name: research-chain
type: chain
agents:
  - researcher
  - summariser
```

### 3. Load and run

```python
factory = AgentFactory(tools)
factory.load("agents/")          # loads every *.yaml in the directory

agent = factory.build("researcher")
result = agent.invoke({"messages": [{"role": "user", "content": "What is LangGraph?"}]})
print(result["messages"][-1].content)
```

Every agent type is invoked identically: `agent.invoke({"messages": [...]})`.

---

## YAML schema

All fields except `name` are optional.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `name` | string | **required** | Used by `factory.build()` |
| `type` | string | `react` | See patterns table above |
| `model` | string | `claude-opus-4-8` | Any Anthropic model ID |
| `system_prompt` | string | `""` | Instructions for the agent |
| `tools` | list | `[]` | Tool names registered in `ToolRegistry` |
| `agents` | list | `[]` | Subagent names (by `name` field) |
| `max_iterations` | int | `5` | Loop cap for `evaluator` |
| `checkpointer` | bool | `false` | Enable in-memory conversation history (`react` only) |

---

## Agent types

### `react`

Single agent that loops: reason → call tool → observe → repeat.

```yaml
name: researcher
type: react
system_prompt: Search the web and return concise, cited findings.
tools:
  - web_search
  - web_fetch
```

---

### `chain` — Prompt chaining

Agents run in order. Each step receives the previous step's output.

```
researcher → summariser → formatter
```

```yaml
name: research-pipeline
type: chain
agents:
  - researcher
  - summariser
  - formatter
```

---

### `router` — Routing

LLM reads the input, picks one agent, hands off, done. The classifier uses each agent's `system_prompt` as its description.

```
input → [classify] → researcher
                   ↘ coder
                   ↘ writer
```

```yaml
name: smart-router
type: router
agents:
  - researcher
  - coder
  - writer
```

---

### `parallel` — Parallelization

All agents receive the same input concurrently. Outputs are merged into one response.

```
         ┌→ analyst-a ─┐
input ───┤              ├→ merged output
         └→ analyst-b ─┘
```

```yaml
name: dual-analysis
type: parallel
agents:
  - analyst-a
  - analyst-b
```

---

### `orchestrator` — Orchestrator-subagents

An orchestrator LLM dynamically decides what to do at each step — unlike `router`, it stays in the loop across multiple turns and synthesises the final result.

> Requires `pip install langgraph-supervisor`

```yaml
name: research-team
type: orchestrator
system_prompt: |
  Coordinate the team. Use researcher for web lookups,
  coder for calculations, writer for the final answer.
agents:
  - researcher
  - coder
  - writer
```

---

### `evaluator` — Evaluator-optimizer

`agents[0]` generates output. `agents[1]` reviews it, replying `ACCEPTED` or `REJECTED: <feedback>`. On rejection the feedback is added to the conversation and the generator retries. Exits on acceptance or `max_iterations`.

```
START → generate → evaluate → ACCEPTED? → END
            ↑                ↘ feedback ↗
```

```yaml
name: write-and-review
type: evaluator
agents:
  - writer    # [0] generator
  - critic    # [1] evaluator
max_iterations: 3
```

The evaluator's `system_prompt` should instruct it to reply `ACCEPTED` or `REJECTED: <specific feedback>`.

---

### `swarm`

Each agent can hand off to any other. The factory wires the handoff tools automatically. Control starts with `agents[0]`.

> Requires `pip install langgraph-swarm`

```yaml
name: research-swarm
type: swarm
agents:
  - researcher
  - summariser
```

---

## Custom workflows

Register any hand-built `StateGraph` directly:

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph = StateGraph(State)
graph.add_node("step", lambda s: {"messages": [{"role": "assistant", "content": "done"}]})
graph.add_edge(START, "step")
graph.add_edge("step", END)

factory.register_graph("my-workflow", graph.compile())
agent = factory.build("my-workflow")
```

---

## Built-in tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web via DuckDuckGo — returns titles, URLs, snippets |
| `web_fetch` | Fetch a URL and extract readable text (strips nav, scripts, boilerplate) |
| `run_python` | Execute Python code and return its stdout |

```python
from engine.tools import web_search, web_fetch, run_python
tools.register(web_search, web_fetch, run_python)
```

---

## Project layout

```
agents/          YAML agent definitions — add a file, get an agent
engine/
  factory.py     AgentFactory — load(), build(), register_graph()
  registry.py    ToolRegistry — register(), .tool decorator
  specs.py       AgentSpec — Pydantic model (YAML maps directly to this)
  tools.py       Built-in tools
  builders/      One builder per pattern, shared helpers in _base.py
main.py          Demo — tools + factory.load() + invocations, nothing else
```
