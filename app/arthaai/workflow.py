from __future__ import annotations
from datetime import datetime, timezone
import json, uuid
from .llm import get_llm
from .data import data_manifest, load_cpi
from .forecast import forecast
from .scenario import simulate
from .schemas import ResearchRequest, ResearchRun, AgentEvent, ScenarioInput

class Workflow:
    allowed_tools={"data":["load_cpi","data_manifest"],"inflation":["load_cpi"],"growth":[],"external":[],"econometrics":["forecast"],"scenario":["simulate"],"policy":["scorecard"],"critic":["validate"],"report":["render"]}
    def __init__(self): self.runs={}
    def run(self, request: ResearchRequest) -> ResearchRun:
        run_id=str(uuid.uuid4()); events=[]; agents={}; errors=[]
        def event(node,status,message,tool=None): events.append(AgentEvent(run_id=run_id,node=node,status=status,message=message,tool=tool,ts=datetime.now(timezone.utc)))
        def node(name, fn):
            event(name,"started",f"{name} started")
            try:
                out=fn(); agents[name]=out; event(name,"completed",f"{name} completed"); return out
            except Exception as e:
                errors.append(f"{name}: {e}"); event(name,"failed","tool execution failed safely"); return None
        plan=node("planner",lambda:get_llm().plan(request.question)) or {"objective":request.question,"agents":[]}
        df=node("data",lambda:{"manifest":data_manifest(),"latest":load_cpi().tail(3).to_dict("records")})
        node("inflation",lambda:{"status":"completed","latest_cpi_yoy":float(load_cpi().cpi_yoy.iloc[-1]),"note":"Derived from compatible CPI index levels."})
        node("growth",lambda:{"status":"not_available","reason":"No activity series was required for this offline run."})
        node("external",lambda:{"status":"not_available","reason":"External adapters are configured separately; scenario inputs are explicit assumptions."})
        fc=node("econometrics",lambda:forecast(request.horizon_months,"best"))
        sc=None
        if request.scenario or plan.get("requires_scenario"):
            sc=node("scenario",lambda:simulate(request.scenario or ScenarioInput(horizon_months=request.horizon_months)))
            node("policy",lambda:sc.policy_scorecard if sc else [])
        critic=node("critic",lambda:self._critic(fc,sc,errors))
        report=node("report",lambda:self._report(request,fc,sc,critic,errors))
        status="completed" if critic and critic.get("status")=="PASS" else "completed_with_warnings"
        audit={"run_id":run_id,"tool_calls":[{"node":e.node,"status":e.status,"tool":e.tool} for e in events],"errors":errors,"llm":get_llm().health(),"version":"1.0.0"}
        run=ResearchRun(run_id=run_id,question=request.question,status=status,plan=plan,forecast=fc,scenario=sc,agents=agents,events=events,report=report,audit=audit)
        self.runs[run_id]=run.model_dump(mode="json"); return run
    def _critic(self,fc,sc,errors):
        issues=[]
        if fc is None: issues.append("Forecast tool did not return a result")
        if fc and any(p.lower80>p.point or p.point>p.upper80 for p in fc.points): issues.append("Forecast interval ordering invalid")
        if errors: issues.extend(errors)
        return {"status":"FAIL" if issues else "PASS","issues":issues,"unsupported_claims":[],"numeric_grounding":"PASS" if fc else "FAIL","retry_count":0}
    def _report(self,req,fc,sc,critic,errors):
        return {"title":"ArthaAI conditional macro research report","question":req.question,"executive_summary":"Deterministic forecast and scenario outputs are shown with provenance. This is a research prototype, not an official RBI forecast.","forecast":fc.model_dump(mode="json") if fc else None,"scenario":sc.model_dump(mode="json") if sc else None,"critic":critic,"limitations":["Historical packaged data lacks a complete real-time vintage panel.","External macro series are not silently fabricated when unavailable.","Scenario pass-through values are illustrative assumptions, not causal estimates."],"sources":["https://www.mospi.gov.in/"],"errors":errors}

workflow=Workflow()
