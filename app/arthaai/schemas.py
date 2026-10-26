from __future__ import annotations
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict

class ScenarioInput(BaseModel):
    oil_pct: float = Field(0, ge=-80, le=300)
    fx_pct: float = Field(0, ge=-80, le=300, description="INR/USD percentage change; positive means depreciation")
    food_pp: float = Field(0, ge=-20, le=20)
    policy_bp: int = Field(0, ge=-200, le=300)
    horizon_months: int = Field(6, ge=1, le=12)
    seed: int = 42
    draws: int = Field(500, ge=100, le=5000)

class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str
    value: float | None = None
    unit: str | None = None
    date: str | None = None
    source: str
    source_url: str
    series_id: str | None = None
    status: Literal["observed", "derived", "assumption", "model_output"]

class ForecastPoint(BaseModel):
    horizon: int
    period: str
    point: float
    lower80: float
    upper80: float

class ForecastResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    target: str = "cpi_yoy"
    model: str
    model_version: str
    data_vintage: str
    trained_until: str
    points: list[ForecastPoint]
    metrics: dict[str, float] = Field(default_factory=dict)
    interval_method: str
    warnings: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

class ScenarioResult(BaseModel):
    run_id: str
    baseline: ForecastResult
    shocked_points: list[ForecastPoint]
    quantiles: dict[str, list[float]]
    assumptions: list[str]
    method: str
    policy_scorecard: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)

class AgentEvent(BaseModel):
    run_id: str
    node: str
    status: Literal["started", "completed", "failed"]
    message: str
    ts: datetime
    tool: str | None = None

class ResearchRequest(BaseModel):
    question: str = Field(min_length=10, max_length=4000)
    scenario: ScenarioInput | None = None
    horizon_months: int = Field(6, ge=1, le=12)

class ResearchRun(BaseModel):
    run_id: str
    question: str
    status: str
    plan: dict[str, Any]
    forecast: ForecastResult | None = None
    scenario: ScenarioResult | None = None
    agents: dict[str, Any]
    events: list[AgentEvent]
    report: dict[str, Any] | None = None
    audit: dict[str, Any]
