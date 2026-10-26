from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "app/models/India_CPI_Models_and_Data/india_cpi_forecasting_package/data/processed/india_headline_cpi_target.csv"
SOURCE_URL = "https://www.mospi.gov.in/"

def load_cpi() -> pd.DataFrame:
    if not TARGET.exists():
        raise FileNotFoundError(f"Packaged CPI target not found: {TARGET}")
    df = pd.read_csv(TARGET)
    df["period"] = pd.to_datetime(df["period"], format="%Y-%m")
    idx = "headline_cpi_index_2012_100_final"
    if idx not in df:
        raise ValueError("CPI index column missing")
    df["cpi_yoy"] = df[idx].pct_change(12) * 100
    df = df.dropna(subset=["cpi_yoy"]).sort_values("period").reset_index(drop=True)
    if len(df) < 36:
        raise ValueError("At least 36 derived CPI observations are required")
    return df[["period", "cpi_yoy", idx, "index_base"]]

def data_manifest() -> dict:
    df = load_cpi()
    raw = TARGET.read_bytes()
    return {
        "series_id": "cpi_yoy",
        "source": "MoSPI package snapshot",
        "source_url": SOURCE_URL,
        "first_period": df.period.min().strftime("%Y-%m"),
        "last_period": df.period.max().strftime("%Y-%m"),
        "rows": int(len(df)),
        "base_year": str(df.index_base.iloc[-1]),
        "checksum": hashlib.sha256(raw).hexdigest(),
        "vintage_note": "The packaged history contains publication metadata but not a complete real-time vintage panel; backtests are historical-snapshot, not vintage-perfect.",
    }

def latest_evidence(df: pd.DataFrame) -> list[dict]:
    row = df.iloc[-1]
    return [{
        "claim": "Derived headline CPI year-on-year inflation",
        "value": float(row.cpi_yoy), "unit": "% YoY", "date": row.period.strftime("%Y-%m"),
        "source": "MoSPI package snapshot", "source_url": SOURCE_URL,
        "series_id": "cpi_yoy", "status": "derived"
    }]
