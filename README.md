# LangGraph Agent Factory

A YAML-driven agent factory built on LangGraph that rigidly follows Anthropic's [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) patterns.

**You define your entire agentic system in a single YAML file.** The factory handles wiring the graph, the LLM, and the tools.

---

## Two categories, one file

Anthropic draws a clear line between two things:

| Category | Who controls flow | Node type |
|----------|------------------|-----------|
| **Agents** | The LLM — it decides what to do next | `react` |
| **Workflows** | Your code — fixed, deterministic paths | `llm` nodes |

This factory keeps that line strict:

- **`react`** — a ReAct agent. The LLM loops: think → call tool → observe → repeat. Use when the agent needs to take actions and decide when it's done.
- **`llm`** — a single LLM call. One invocation, returns, done. The building block for all workflow steps.

Workflow types (`chain`, `router`, `parallel`, `orchestrator`, `evaluator`) compose `llm` and `react` nodes into a fixed graph topology.

---

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python main.py
```

---

## Everything in one YAML file

### Standalone agent (`react`)

```yaml
name: web-researcher
type: react
system_prompt: Search the web to answer questions thoroughly. Always cite URLs.
tools:
  - web_search
  - web_fetch
```

### Chain — prompt chaining

```yaml
name: research-pipeline
type: chain
steps:
  - name: researcher
    type: react
    system_prompt: Search the web and gather comprehensive information.
    tools:
      - web_search
      - web_fetch

  - name: summariser
    type: llm
    system_prompt: Condense the research into clear bullet points.

  - name: writer
    type: llm
    system_prompt: Rewrite the summary as polished, publication-ready prose.
```

### Router — routing

```yaml
name: smart-router
type: router
steps:
  - name: researcher
    type: react
    system_prompt: Answer factual and web research questions.
    tools:
      - web_search
      - web_fetch

  - name: analyst
    type: react
    system_prompt: Answer calculation and coding questions by running Python.
    tools:
      - run_python

  - name: writer
    type: llm
    system_prompt: Answer writing, editing, and summarisation tasks.
```

### Parallel — parallelization

```yaml
name: parallel-perspectives
type: parallel
steps:
  - name: technical
    type: llm
    system_prompt: Analyse the topic from a technical engineering perspective.

  - name: business
    type: llm
    system_prompt: Analyse the topic from a business and market perspective.
```

### Evaluator — evaluator-optimizer

```yaml
name: write-and-review
type: evaluator
max_iterations: 3
steps:
  - name: writer
    type: llm
    system_prompt: Write a clear, accurate response to the request.

  - name: critic
    type: llm
    system_prompt: |
      Review the response. Reply ACCEPTED if it is clear and complete,
      or REJECTED: <specific feedback> if it needs improvement.
```

### Orchestrator — orchestrator-subagents

```yaml
name: research-team
type: orchestrator
system_prompt: |
  Coordinate your team to answer the question, then synthesise a final response.
steps:
  - name: researcher
    type: react
    system_prompt: Search the web for information. Cite sources.
    tools:
      - web_search
      - web_fetch

  - name: analyst
    type: react
    system_prompt: Write and run Python to compute or process data.
    tools:
      - run_python

  - name: writer
    type: llm
    system_prompt: Write the final polished answer.
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

agent = factory.build("research-pipeline")
result = agent.invoke({"messages": [{"role": "user", "content": "What is LangGraph?"}]})
print(result["messages"][-1].content)
```

Every agent type is invoked identically: `agent.invoke({"messages": [...]})`.

---

## Built-in tools

| Tool | What it does |
|------|-------------|
| `web_search` | Search the web via DuckDuckGo |
| `web_fetch` | Fetch a URL and extract readable text |
| `bash` | Execute a shell command |
| `run_python` | Execute Python code via `python3 -c` |

```python
from engine.tools import BUILTIN_TOOLS
tools.register(*BUILTIN_TOOLS)
```

Add your own:

```python
@tools.tool
def word_count(text: str) -> str:
    """Count the words in a text string."""
    return f"{len(text.split())} words"
```

---

## YAML schema

| Field | Type | Default | Used by |
|-------|------|---------|---------|
| `name` | string | **required** | all |
| `type` | string | `react` | all |
| `model` | string | `claude-opus-4-8` | all |
| `system_prompt` | string | `""` | all |
| `tools` | list | `[]` | `llm`, `react` |
| `steps` | list | `[]` | workflow types |
| `max_iterations` | int | `5` | `evaluator` |
| `checkpointer` | bool | `false` | `react` |

Each item in `steps` is a full inline node spec (same fields, `type` must be `llm` or `react`).

---

## Project layout

```
agents/          YAML files — one file per agentic system
engine/
  factory.py     AgentFactory — load(), build(), register_graph()
  registry.py    ToolRegistry — register(), .tool decorator
  specs.py       AgentSpec — Pydantic model, YAML maps directly to this
  tools.py       Built-in tools + BUILTIN_TOOLS list
  builders/
    _base.py     BaseBuilder + MessagesState
    llm.py       Single LLM call
    react.py     ReAct agent loop
    chain.py     Prompt chaining workflow
    router.py    Routing workflow
    parallel.py  Parallelization workflow
    orchestrator.py  Orchestrator-subagents workflow
    evaluator.py     Evaluator-optimizer workflow
main.py          Demo
```
