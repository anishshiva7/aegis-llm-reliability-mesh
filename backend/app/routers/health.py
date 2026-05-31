"""
/providers/health endpoint — per-provider reliability view (Module 7).

Exposes the rolling health the fallback chain uses to decide ordering: which
providers are healthy / degraded / unhealthy, their failure streaks, and the
recommended (healthiest-first, mock-last) attempt order.
"""

from fastapi import APIRouter

from ..logging_config import get_logger
from ..models.schemas import ProviderHealth, ProvidersHealthResponse
from ..services.health import get_health_registry

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/providers/health", response_model=ProvidersHealthResponse)
def providers_health() -> ProvidersHealthResponse:
    """Return each provider's health plus the health-aware fallback order."""
    registry = get_health_registry()
    snap = registry.snapshot()
    providers = [ProviderHealth(**row) for row in snap]
    order = registry.order([row["provider"] for row in snap])
    return ProvidersHealthResponse(providers=providers, recommended_order=order)
