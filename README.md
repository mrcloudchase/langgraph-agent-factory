# LangGraph Agent Factory

A YAML-driven agent factory that implements every agentic system pattern from Anthropic's [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) using LangGraph.

**The factory provides**: seven agentic patterns as code and four built-in tools.  
**You provide**: YAML files that define what your agents do, which pattern they use, and which tools they have.

No Python required to author agents. Write YAML, get a running agentic system.

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

## How it works

Everything is YAML. Write a file in `agents/`, run `factory.load("agents/")`, and call `factory.build("your-agent-name")`. That's it.

The factory wires the pattern (graph topology), the LLM, and the tools together from your spec. You focus on what the agent does — not how the graph is built.

---

## Built-in tools

Four tools are available to any agent by name:

| Tool | What it does |
|------|-------------|
| `web_search` | Search the web via DuckDuckGo — returns titles, URLs, snippets |
| `web_fetch` | Fetch a URL and extract readable text (strips nav, scripts, boilerplate) |
| `bash` | Execute a shell command and return its output |
| `run_python` | Execute Python code via `python3 -c` in an isolated subprocess |

Register all four at once:

```python
from engine.tools import BUILTIN_TOOLS
tools.register(*BUILTIN_TOOLS)
```

Or pick and choose:

```python
from engine.tools import web_search, run_python
tools.register(web_search, run_python)
```

You can also register your own tools:

```python
@tools.tool
def word_count(text: str) -> str:
    """Count the words in a text string."""
    return f"{len(text.split())} words"
```

---

## Authoring agents in YAML

Create a file in `agents/` for each agent. The `name` field is what you pass to `factory.build()`.

### Leaf agents (do the actual work)

```yaml
# agents/researcher.yaml
name: researcher
type: react
system_prompt: |
  You are a research assistant. Search the web to answer questions thoroughly.
  Always cite your sources with URLs.
tools:
  - web_search
  - web_fetch
```

```yaml
# agents/coder.yaml
name: coder
type: react
system_prompt: |
  You are a Python coding assistant. Write and run code to verify your answers.
tools:
  - run_python
```

### Composite agents (orchestrate leaf agents)

Composite agents reference other agents by their `name` field. Load order doesn't matter — graphs are built on demand.

```yaml
# agents/research-chain.yaml  (prompt chaining)
name: research-chain
type: chain
agents:
  - researcher   # step 1: find information
  - summariser   # step 2: distill findings
  - writer       # step 3: polish the prose
```

```yaml
# agents/smart-router.yaml  (routing)
name: smart-router
type: router
agents:
  - researcher   # handles factual / web questions
  - coder        # handles calculations and code
  - writer       # handles writing and editing tasks
```

```yaml
# agents/research-team.yaml  (orchestrator)
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

```yaml
# agents/write-and-review.yaml  (evaluator)
name: write-and-review
type: evaluator
max_iterations: 3
agents:
  - writer    # [0] generator
  - critic    # [1] evaluator — must reply ACCEPTED or REJECTED: <feedback>
```

---

## Loading and running

```python
from engine import AgentFactory, ToolRegistry
from engine.tools import BUILTIN_TOOLS

tools = ToolRegistry()
tools.register(*BUILTIN_TOOLS)

factory = AgentFactory(tools)
factory.load("agents/")          # loads every *.yaml in the directory

agent = factory.build("research-chain")
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
researcher → summariser → writer
```

```yaml
name: research-pipeline
type: chain
agents:
  - researcher
  - summariser
  - writer
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
         ┌→ researcher ─┐
input ───┤               ├→ merged output
         └→ coder ───────┘
```

```yaml
name: parallel-team
type: parallel
agents:
  - researcher
  - coder
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

## Project layout

```
agents/          YAML agent definitions — add a file, get an agent
engine/
  factory.py     AgentFactory — load(), build(), register_graph()
  registry.py    ToolRegistry — register(), .tool decorator
  specs.py       AgentSpec — Pydantic model (YAML maps directly to this)
  tools.py       Built-in tools (web_search, web_fetch, bash, run_python)
  builders/      One builder per pattern, shared helpers in _base.py
main.py          Demo — tools + factory.load() + invocations
```
