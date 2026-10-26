from __future__ import annotations
import math, uuid
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .data import load_cpi, data_manifest, latest_evidence
from .schemas import ForecastPoint, ForecastResult

@dataclass
class BacktestRow:
    model: str
    horizon: int
    mae: float
    rmse: float
    bias: float
    n: int

def _seasonal(y: np.ndarray, h: int) -> np.ndarray:
    if len(y) >= 12: return np.repeat(y[-12], h)
    return np.repeat(y[-1], h)

def _persistence(y: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(y[-1], h)

def _ar1(y: np.ndarray, h: int) -> np.ndarray:
    if len(y) < 24: return _persistence(y, h)
    x, z = y[:-1], y[1:]
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, z, rcond=None)[0]
    out=[]; prev=float(y[-1])
    for _ in range(h):
        prev=float(beta[0] + beta[1]*prev); out.append(prev)
    return np.array(out)

def predict_array(y: np.ndarray, h: int, model: str) -> np.ndarray:
    if model == "seasonal_naive": return _seasonal(y, h)
    if model == "persistence": return _persistence(y, h)
    if model in {"ar1", "arima"}: return _ar1(y, h)
    raise ValueError(f"Unsupported model: {model}")

def backtest(horizon: int = 1, min_train: int = 36) -> list[BacktestRow]:
    df=load_cpi(); y=df.cpi_yoy.to_numpy(float)
    rows=[]
    for model in ["seasonal_naive", "persistence", "ar1"]:
        actual=[]; pred=[]
        for origin in range(min_train, len(y)-horizon+1):
            p=predict_array(y[:origin], horizon, model)[-1]
            actual.append(y[origin+horizon-1]); pred.append(p)
        err=np.asarray(pred)-np.asarray(actual)
        rows.append(BacktestRow(model,horizon,float(np.mean(np.abs(err))),float(np.sqrt(np.mean(err**2))),float(np.mean(err)),len(err)))
    return rows

def forecast(horizon: int = 6, model: str = "ar1") -> ForecastResult:
    df=load_cpi(); y=df.cpi_yoy.to_numpy(float); manifest=data_manifest()
    if model == "best":
        candidates=backtest(min(horizon, 6)); model=min(candidates, key=lambda r:r.mae).model
    selected=model
    values=predict_array(y, horizon, selected)
    bt=next((r for r in backtest(min(horizon,6)) if r.model==selected), None)
    sigma=float(max(0.15, bt.rmse if bt else np.std(np.diff(y))))
    last=df.period.iloc[-1]
    points=[]
    for i,v in enumerate(values,1):
        period=(last + pd.offsets.MonthBegin(i)).strftime("%Y-%m")
        points.append(ForecastPoint(horizon=i, period=period, point=round(float(v),4), lower80=round(float(v-1.28*sigma*math.sqrt(i)),4), upper80=round(float(v+1.28*sigma*math.sqrt(i)),4)))
    return ForecastResult(run_id=str(uuid.uuid4()), model=selected, model_version="arthaai-deterministic-1.0", data_vintage=manifest["last_period"], trained_until=manifest["last_period"], points=points, metrics={"mae": bt.mae if bt else 0.0, "rmse": bt.rmse if bt else 0.0, "bias": bt.bias if bt else 0.0, "n": bt.n if bt else 0}, interval_method="residual RMSE scaled by sqrt(h); empirical coverage must be recalibrated on refreshed vintages", warnings=[manifest["vintage_note"]], evidence=latest_evidence(df))
