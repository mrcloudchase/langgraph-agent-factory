"""Runtime — executes a ServiceSpec and returns the result.

The runtime is intentionally thin. It takes a ServiceSpec,
assembles the right agent, runs it, handles delivery, and returns
a ServiceRun. It knows nothing about how the spec was created.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from .models import ServiceRun, ServiceSpec
from .tools import TOOL_REGISTRY


class Runtime:
    def __init__(self, model: str = "claude-opus-4-8"):
        self._model = model

    def run(self, spec: ServiceSpec, customer_id: str, input_text: str) -> ServiceRun:
        """Execute a ServiceSpec and return a ServiceRun record."""
        tools = [TOOL_REGISTRY[t] for t in spec.tools if t in TOOL_REGISTRY]

        agent = create_react_agent(
            model=ChatAnthropic(model=self._model),
            tools=tools,
            prompt=spec.system_prompt,
        )

        result = agent.invoke({
            "messages": [{"role": "user", "content": input_text}]
        })
        output = result["messages"][-1].content

        # If the spec has a real delivery destination, push the output there.
        if spec.output_destination not in ("return", ""):
            deliver = TOOL_REGISTRY.get("deliver_output")
            if deliver:
                deliver.invoke({
                    "destination": spec.output_destination,
                    "content": output,
                })

        spec.run_count += 1

        return ServiceRun(
            service_id=spec.id,
            customer_id=customer_id,
            input=input_text,
            output=output,
            cost=spec.price_per_run,
        )
