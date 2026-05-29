"""MVP demo: deliver a managed-services outcome through the agent factory.

    python main.py

Requires ANTHROPIC_API_KEY in the environment (see .env.example).
"""

import os
import sys

from factory import AgentFactory, OutcomeRunner


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY (see .env.example) before running.")

    factory = AgentFactory()
    runner = OutcomeRunner(factory, "catalog/outcomes.yaml")

    print("=== Delivering outcome: incident_resolve_p3_p4 ===\n")
    summary, charge = runner.deliver(
        outcome_id="incident_resolve_p3_p4",
        customer_id="contoso",
        trigger=(
            "EM7 alert ALRT-9981 fired: high CPU on web-prod-04. "
            "Triage it, open an incident, remediate, and resolve."
        ),
    )

    print(summary)
    print("\n--- Billing ---")
    print(f"{charge['outcome_name']}: ${charge['amount']:.2f}  "
          f"(SLA met: {charge['sla_met']})")
    print(f"Session revenue: ${runner.revenue():.2f}")


if __name__ == "__main__":
    main()
