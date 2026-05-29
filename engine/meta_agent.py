"""MetaAgent — turns any natural language service description into a ServiceSpec.

This is the core of the platform. The customer describes what they want.
The MetaAgent figures out what tools, prompt, trigger, and pricing are needed,
then returns a ServiceSpec that the Runtime can execute immediately.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from .models import ServiceSpec

_SYSTEM = """\
You are the service designer for an AI-powered outcomes platform.

Customers describe a desired outcome in plain language. You turn their
description into a ServiceSpec — a precise, self-contained blueprint that
an autonomous agent will execute to deliver that outcome.

━━━ Available tools ━━━
  web_search      Search the web for current information
  browse_url      Fetch and read content from a specific URL
  run_python      Execute Python for data processing or formatting
  deliver_output  Send the final result to the customer's destination

━━━ Design principles ━━━
  • system_prompt must be fully self-contained. The agent that runs
    this service has no memory of this conversation. Every decision,
    format requirement, and edge case must be encoded in the prompt.
  • One service = one clear outcome. Keep scope tight.
  • Price fairly:
      web research, simple summaries    → $2–8
      multi-source analysis             → $8–20
      complex multi-step workflows      → $20–50
  • If no schedule is mentioned, use trigger: on_demand
  • If no delivery destination is mentioned, use output_destination: return
"""

_CLARIFY = """\
A customer wants to use the platform. Their request:

{request}

If you have enough information to build the service, reply with exactly: READY

Otherwise ask 1–3 short, specific clarifying questions.
No preamble — just the questions.
"""


class MetaAgent:
    def __init__(self, model: str = "claude-opus-4-8"):
        llm = ChatAnthropic(model=model, temperature=0)
        self._chat = llm
        self._designer = llm.with_structured_output(ServiceSpec)

    def clarify(self, request: str) -> str:
        """Return 'READY' if enough context exists, or clarifying questions."""
        msg = self._chat.invoke([
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _CLARIFY.format(request=request)},
        ])
        return msg.content.strip()

    def design(self, request: str, extra_context: str = "") -> ServiceSpec:
        """Produce a ServiceSpec from a natural language service description."""
        content = request
        if extra_context:
            content += f"\n\nAdditional context: {extra_context}"
        return self._designer.invoke([
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": f"Design a service for: {content}"},
        ])
