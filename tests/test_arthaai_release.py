from app.arthaai.data import load_cpi, data_manifest
from app.arthaai.forecast import forecast, backtest
from app.arthaai.scenario import simulate
from app.arthaai.schemas import ScenarioInput, ResearchRequest
from app.arthaai.workflow import Workflow
from app.arthaai.audit import audit_run

def test_data_manifest_and_yoy():
    df=load_cpi(); assert len(df)>=36; assert df.cpi_yoy.notna().all(); assert data_manifest()["checksum"]

def test_forecast_contract_and_intervals():
    out=forecast(3,"best"); assert len(out.points)==3; assert all(p.lower80<=p.point<=p.upper80 for p in out.points); assert out.evidence

def test_backtest_has_baselines():
    rows=backtest(1); assert {r.model for r in rows}=={"seasonal_naive","persistence","ar1"}; assert all(r.n>0 for r in rows)

def test_scenario_units_and_scorecard():
    out=simulate(ScenarioInput(oil_pct=20,fx_pct=5,food_pp=1.5,policy_bp=25,horizon_months=6,draws=100)); assert len(out.shocked_points)==6; assert len(out.policy_scorecard)==4; assert len(out.quantiles["shock_impact_pp_p10_p50_p90"])==3

def test_end_to_end_workflow_and_audit():
    run=Workflow().run(ResearchRequest(question="Forecast CPI with an oil shock and policy scenarios.",scenario=ScenarioInput(oil_pct=20,fx_pct=5,food_pp=1.5,horizon_months=3,draws=100))); assert run.status.startswith("completed"); assert run.forecast; assert run.report; assert audit_run(run.model_dump(mode="json"))["status"]=="PASS"
