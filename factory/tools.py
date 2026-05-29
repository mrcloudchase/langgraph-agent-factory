"""Tool registry — the bridge between YAML names and real platform actions.

In production each of these would call ServiceNow / Azure Monitor / EM7 / etc.
For the MVP they return canned data so the whole pipeline runs end-to-end
without any external credentials. Swapping in a real API is a one-function change.
"""

from __future__ import annotations

import random

from langchain_core.tools import tool


@tool
def get_alert(alert_id: str) -> dict:
    """Get full alert details from the monitoring platform (e.g. EM7 / Azure Monitor)."""
    return {
        "alert_id": alert_id,
        "message": "CPU utilization critical: 97% for 15 minutes",
        "severity": "high",
        "device": "web-prod-04",
    }


@tool
def get_device_status(hostname: str) -> dict:
    """Get current health and key metrics for a device."""
    # Pretend the device is unhealthy until a remediation has run this session.
    return {"hostname": hostname, "cpu_percent": 97, "status": "degraded"}


@tool
def create_incident(short_description: str, severity: str) -> dict:
    """Open an incident ticket in the ITSM platform (e.g. ServiceNow)."""
    number = f"INC{random.randint(10000, 99999)}"
    return {"number": number, "state": "new", "short_description": short_description}


@tool
def restart_service(hostname: str, service: str) -> dict:
    """Remediation action: restart a named service on a host (e.g. via Ansible/SSH)."""
    return {"hostname": hostname, "service": service, "result": "restarted", "cpu_percent": 12}


@tool
def update_incident(number: str, state: str, work_notes: str) -> dict:
    """Update an incident's state and append work notes."""
    return {"number": number, "state": state, "work_notes": work_notes}


# name -> callable. YAML files reference tools by these string keys.
TOOL_REGISTRY = {
    fn.name: fn
    for fn in [get_alert, get_device_status, create_incident, restart_service, update_incident]
}
