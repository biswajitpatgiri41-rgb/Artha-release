from __future__ import annotations
from fastapi import APIRouter, HTTPException
from app.arthaai.schemas import ResearchRequest, ScenarioInput
from app.arthaai.data import data_manifest, load_cpi
from app.arthaai.forecast import forecast, backtest
from app.arthaai.scenario import simulate
from app.arthaai.workflow import workflow
from app.arthaai.evaluation import evaluate
from app.arthaai.audit import audit_run
router=APIRouter(prefix="/v1/arthaai", tags=["ArthaAI verified workflow"])
@router.get("/health")
def health():
    try: manifest=data_manifest(); return {"status":"ok","version":"1.0.0","data":manifest,"llm":__import__('app.arthaai.llm',fromlist=['get_llm']).get_llm().health()}
    except Exception as exc: raise HTTPException(503,"ArthaAI data foundation unavailable") from exc
@router.get("/data/manifest")
def manifest(): return data_manifest()
@router.get("/data/cpi")
def cpi(): return {"series_id":"cpi_yoy","observations":load_cpi().to_dict("records")}
@router.get("/forecast")
def get_forecast(horizon:int=6, model:str="best"): return forecast(horizon,model)
@router.get("/backtest")
def get_backtest(horizon:int=1): return {"horizon":horizon,"rows":[r.__dict__ for r in backtest(horizon)]}
@router.post("/simulate")
def post_simulate(request:ScenarioInput): return simulate(request)
@router.post("/research")
def research(request:ResearchRequest): return workflow.run(request)
@router.get("/research/{run_id}")
def get_research(run_id:str):
    if run_id not in workflow.runs: raise HTTPException(404,"Research run not found")
    return workflow.runs[run_id]
@router.get("/research/{run_id}/audit")
def get_audit(run_id:str):
    if run_id not in workflow.runs: raise HTTPException(404,"Research run not found")
    return audit_run(workflow.runs[run_id])
@router.get("/llm/evaluation")
def llm_evaluation(): return evaluate()
