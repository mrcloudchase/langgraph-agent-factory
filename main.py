"""
End-to-end platform demo.

A customer describes a service in plain language.
The meta-agent designs it. The runtime delivers it.
The marketplace stores it for everyone.

    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
    python main.py
"""

from __future__ import annotations

import os
import sys

from engine import Marketplace, MetaAgent, Runtime

# ── The customer's request ───────────────────────────────────────────────────

REQUEST = (
    "Every week give me a digest of what Salesforce, HubSpot, and Notion "
    "announced — blog posts, press releases, product launches. "
    "Just the headlines and a one-line 'so what' for each item."
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def section(label: str) -> None:
    width = 64
    pad = (width - len(label) - 2) // 2
    print(f"\n{'─' * pad} {label} {'─' * pad}\n")


# ── Demo ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY before running.  See .env.example.")

    meta      = MetaAgent()
    runtime   = Runtime()
    market    = Marketplace()

    # 1 ── Customer describes the service ────────────────────────────────────
    section("CUSTOMER REQUEST")
    print(REQUEST)

    # 2 ── Meta-agent checks if it needs clarification ───────────────────────
    section("META-AGENT: CLARIFY")
    answer = meta.clarify(REQUEST)
    if answer == "READY":
        print("Enough information — proceeding to design.")
    else:
        print(answer)
        print("\n(Continuing with what we have for the demo.)")

    # 3 ── Design the service ─────────────────────────────────────────────────
    section("META-AGENT: DESIGN")
    print("Generating service spec...\n")
    spec = meta.design(REQUEST)

    print(f"  Name        {spec.name}")
    print(f"  Description {spec.description}")
    print(f"  Tools       {', '.join(spec.tools)}")
    print(f"  Trigger     {spec.trigger}")
    print(f"  Delivers to {spec.output_destination}")
    print(f"  Price       ${spec.price_per_run:.2f} per run")
    print(f"\n  System prompt (excerpt):")
    print("  " + spec.system_prompt[:300].replace("\n", "\n  ") + "…")

    # 4 ── Execute the service ────────────────────────────────────────────────
    section("RUNTIME: EXECUTE")
    print("Running service...\n")

    run = runtime.run(
        spec,
        customer_id="demo",
        input_text="Run the weekly competitor digest for this week.",
    )

    section("OUTPUT")
    print(run.output)
    print(f"\n  Cost: ${run.cost:.2f}")
    print(f"  Success: {run.success}")

    # 5 ── Publish to marketplace ─────────────────────────────────────────────
    section("MARKETPLACE")
    market.publish(spec, verified=False)

    print(f"  Published:  '{spec.name}'  (id: {spec.id})")
    print(f"  Verified:   {spec.verified}  — needs more runs before going public")
    print(f"  Catalog:    {len(market.list())} service(s) total")
    print()
    print("  Once verified this service is available to every customer")
    print("  on the platform with one-click deploy.")


if __name__ == "__main__":
    main()
