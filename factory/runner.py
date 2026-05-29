"""OutcomeRunner — the business layer that makes this 'services-as-software'.

The factory builds agents; the runner *sells outcomes*. Each catalog entry maps a
priced, SLA-governed outcome to a YAML agent/workflow. Delivering an outcome runs
the agent and records a charge — billing on results, not on time.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import yaml


class OutcomeRunner:
    def __init__(self, factory, catalog_path: str | Path):
        self.factory = factory
        catalog = yaml.safe_load(Path(catalog_path).read_text())
        self.catalog = {o["id"]: o for o in catalog["outcomes"]}
        self.ledger: list[dict] = []

    def deliver(self, outcome_id: str, customer_id: str, trigger: str):
        """Deliver one priced outcome for a customer and record the charge."""
        if outcome_id not in self.catalog:
            raise KeyError(f"No such outcome: {outcome_id}")
        outcome = self.catalog[outcome_id]

        agent = self.factory.from_yaml(outcome["agent"])
        thread_id = f"{customer_id}:{outcome_id}:{uuid4().hex[:8]}"

        started = datetime.utcnow()
        deadline = started + timedelta(hours=outcome.get("sla_hours", 24))

        result = agent.invoke(
            {"messages": [{"role": "user", "content": trigger}]},
            config={"configurable": {"thread_id": thread_id}},
        )

        delivered = datetime.utcnow()
        charge = {
            "customer_id": customer_id,
            "outcome_id": outcome_id,
            "outcome_name": outcome["name"],
            "amount": float(outcome["price"]),
            "currency": outcome.get("currency", "USD"),
            "thread_id": thread_id,
            "sla_met": delivered <= deadline,
            "delivered_at": delivered.isoformat(),
        }
        self.ledger.append(charge)

        summary = result["messages"][-1].content
        return summary, charge

    def revenue(self) -> float:
        return sum(c["amount"] for c in self.ledger)
