"""Marketplace — the catalog of services built by customers.

Services start private (verified=False). After enough successful runs
they can be verified and made available to all customers. The catalog
is crowd-built: the platform never writes a service spec directly.

Persistence is a local JSON file. In production this would be a database.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import ServiceSpec

_CATALOG_PATH = Path("catalog.json")


class Marketplace:
    def __init__(self, catalog_path: Path = _CATALOG_PATH):
        self._path = catalog_path
        self._catalog: dict[str, ServiceSpec] = {}
        self._load()

    # ── write ────────────────────────────────────────────────────────

    def publish(self, spec: ServiceSpec, *, verified: bool = False) -> ServiceSpec:
        """Add or update a service. Unverified by default."""
        spec.verified = verified
        self._catalog[spec.id] = spec
        self._save()
        return spec

    def verify(self, service_id: str) -> ServiceSpec:
        """Mark a service as verified — makes it visible to all customers."""
        spec = self._catalog[service_id]
        spec.verified = True
        self._save()
        return spec

    # ── read ─────────────────────────────────────────────────────────

    def list(self, *, verified_only: bool = False) -> list[ServiceSpec]:
        services = list(self._catalog.values())
        if verified_only:
            return [s for s in services if s.verified]
        return services

    def get(self, service_id: str) -> ServiceSpec | None:
        return self._catalog.get(service_id)

    def search(self, query: str) -> list[ServiceSpec]:
        q = query.lower()
        return [
            s for s in self._catalog.values()
            if q in s.name.lower() or q in s.description.lower()
        ]

    # ── persistence ──────────────────────────────────────────────────

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(
                [s.model_dump() for s in self._catalog.values()],
                indent=2,
            )
        )

    def _load(self) -> None:
        if self._path.exists():
            for item in json.loads(self._path.read_text()):
                spec = ServiceSpec(**item)
                self._catalog[spec.id] = spec
