from __future__ import annotations
from typing import Any

def audit_run(run: dict[str,Any]) -> dict[str,Any]:
    fc=run.get("forecast") or {}; points=fc.get("points") or []
    checks={
        "run_id_present":bool(run.get("run_id")),
        "forecast_values_present":bool(points) and all(isinstance(p.get("point"),(int,float)) for p in points),
        "interval_order":all(p.get("lower80",0)<=p.get("point",0)<=p.get("upper80",0) for p in points),
        "model_version_present":bool(fc.get("model_version")),
        "source_citation_present":bool(fc.get("evidence")),
        "critic_pass_or_warning":(run.get("report") or {}).get("critic",{}).get("status") in {"PASS","FAIL"},
        "tool_events_present":bool((run.get("audit") or {}).get("tool_calls")),
    }
    return {"status":"PASS" if all(checks.values()) else "FAIL","checks":checks}
