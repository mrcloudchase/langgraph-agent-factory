"""Runtime — executes a ServiceSpec and returns the result.

The runtime is intentionally thin. It takes a ServiceSpec,
assembles the right agent, runs it, handles delivery, and returns
a ServiceRun. It knows nothing about how the spec was created.

Delivery is a platform concern owned here, not a tool the agent calls.
The agent produces output; the runtime routes it to the destination.
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

        self._deliver(spec.output_destination, output)
        spec.run_count += 1

        return ServiceRun(
            service_id=spec.id,
            customer_id=customer_id,
            input=input_text,
            output=output,
            cost=spec.price_per_run,
        )

    @staticmethod
    def _deliver(destination: str, content: str) -> None:
        """Route output to the customer's chosen destination.

        In production each branch calls a real integration.
        In the MVP the output is already printed by main.py,
        so non-return destinations just note where they would go.
        """
        if destination in ("return", ""):
            return
        if destination.startswith("email:"):
            # production: send via SES / SendGrid / SMTP
            print(f"\n  → Email {destination[6:]}")
        elif destination.startswith("slack:"):
            # production: post via Slack Web API
            print(f"\n  → Slack {destination[6:]}")
        elif destination.startswith("webhook:"):
            # production: HTTP POST to destination[8:]
            print(f"\n  → Webhook {destination[8:]}")
        else:
            print(f"\n  → {destination}")
