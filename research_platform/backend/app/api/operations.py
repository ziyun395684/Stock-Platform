"""Operational health and notification helpers."""
from fastapi import APIRouter, HTTPException

from ..operations import check_data_freshness
from ..schemas import DataFreshnessRequest

router = APIRouter(prefix="/operations", tags=["operations"])


@router.post("/data-freshness")
def data_freshness(payload: DataFreshnessRequest) -> dict:
    try:
        return check_data_freshness(payload.latest_date, payload.expected_date, max_lag_days=payload.max_lag_days)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
