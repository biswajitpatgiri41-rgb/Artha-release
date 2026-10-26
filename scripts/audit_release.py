import json
from pathlib import Path
from app.arthaai.workflow import workflow
from app.arthaai.schemas import ResearchRequest, ScenarioInput
from app.arthaai.audit import audit_run
run=workflow.run(ResearchRequest(question="Forecast CPI for six months with an oil shock and compare policy scenarios.",scenario=ScenarioInput(oil_pct=20,fx_pct=5,food_pp=1.5,horizon_months=6)))
result=audit_run(run.model_dump(mode="json")); result["run_id"]=run.run_id; Path("reports").mkdir(exist_ok=True); Path("reports/release_audit.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
