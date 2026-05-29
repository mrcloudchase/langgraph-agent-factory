from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


class ServiceSpec(BaseModel):
    """Everything needed to build and run an agentic service."""

    name: str = Field(description="Short name for the service (3–5 words)")
    description: str = Field(description="One sentence: what outcome this service delivers")
    system_prompt: str = Field(
        description=(
            "Full instructions for the agent that will execute this service. "
            "Must be self-contained — the agent runs without any human in the loop. "
            "Encode every decision, format, and edge case here."
        )
    )
    tools: list[str] = Field(
        description=(
            "Tools the agent needs. "
            "Available: web_search, browse_url, run_python, deliver_output"
        )
    )
    trigger: str = Field(
        description=(
            "When the service runs. "
            "Use 'on_demand' or a schedule like 'weekly:MON:08:00' / 'daily:07:00'"
        )
    )
    output_destination: str = Field(
        description=(
            "Where to send results. "
            "Use 'return' to return inline, "
            "'email:address@domain.com', or 'slack:#channel-name'"
        )
    )
    price_per_run: float = Field(
        description="Fair price in USD per delivery. Simple research: $2–8. Complex: $10–50."
    )

    # Platform-managed fields — Claude should not need to fill these.
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    creator_id: str = "platform"
    verified: bool = False
    run_count: int = 0
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ServiceRun(BaseModel):
    """Record of one execution of a service."""

    service_id: str
    customer_id: str
    input: str
    output: str
    cost: float
    success: bool = True
    ran_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
