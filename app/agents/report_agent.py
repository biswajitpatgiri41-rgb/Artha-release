"""Report agent.

The report agent synthesises validated agent outputs into a final
ReportResult.  It only uses evidence that is present and does not
introduce new economic numbers.
"""

from __future__ import annotations

from typing import List

from .schemas import ReportResult, AgentAnalysis, PolicyAnalysis


def run_report(
    planner: dict,
    data: dict,
    growth: AgentAnalysis,
    external: AgentAnalysis,
    econometrics: dict,
    policy: PolicyAnalysis,
    critic: dict,
    retrieved_evidence: List[dict] | None = None,
):
    """Generate a report from the validated outputs.

    Parameters are raw dictionaries or Pydantic models from the agents.
    """

    # Aggregate uncertainties
    uncertainties: List[str] = []
    for obj in [growth, external, policy]:
        uncertainties.extend(getattr(obj, "uncertainties", []))
        uncertainties.extend(getattr(obj, "warnings", []))
    uncertainties.extend(critic.get("issues", []))

    sources: List[str] = []
    # Collect source names from evidence
    for evidence in growth.evidence + external.evidence + policy.evidence:
        if evidence.source:
            sources.append(evidence.source)
    evidence_items = [item.model_dump() if hasattr(item, "model_dump") else item for item in (retrieved_evidence or [])]
    document_labels = [f"{item['filename']} (doc)" for item in evidence_items if item.get("filename")]
    sources = list(dict.fromkeys(sources + document_labels))

    model_evidence = []
    if econometrics:
        model_evidence.append({"type": "MODEL OUTPUT", "model": econometrics.get("model") or econometrics.get("model_name"),
                               "model_version": econometrics.get("model_version"), "output": econometrics,
                               "note": "Numerical forecast output; document retrieval is not forecast validation."})
    document_evidence = []
    for item in evidence_items:
        location = item.get("location", {})
        reference = {"filename": item.get("filename"), "format": item.get("format"), "evidence_id": item.get("evidence_id"),
                     "excerpt": item.get("excerpt"), "structured_result": item.get("structured_result")}
        if location.get("page") is not None:
            reference["page"] = location["page"]
        if location.get("sheet") is not None:
            reference["sheet"] = location["sheet"]
        if location.get("rows") is not None:
            reference["rows"] = location["rows"]
        if location.get("columns") is not None:
            reference["columns"] = location["columns"]
        if location.get("path") is not None:
            reference["path"] = location["path"]
        if location.get("section") is not None:
            reference["section"] = location["section"]
        document_evidence.append(reference)

    citation_summary = ""
    if retrieved_evidence:
        refs = []
        for item in evidence_items:
            location = item.get("location", {})
            if item.get("filename") and location.get("page") is not None:
                refs.append(f"{item['filename']} page {location['page']}")
        if refs:
            citation_summary = " Supporting document evidence: " + "; ".join(refs) + "."
    return ReportResult(
        executive_summary="Prototype scenario analysis. Not an official RBI recommendation." + citation_summary,
        current_state="Data currently unavailable. All analyses are provisional.",
        inflation_view="No quantitative inflation data available.",
        growth_view="Growth assessment is qualitative.",
        external_view="External factors are qualitative.",
        scenario_view="Scenario details are based on user input.",
        policy_view="Policy options are conceptual.",
        uncertainties=uncertainties,
        sources=sources,
        disclaimer="Research prototype. Not an official RBI forecast or policy recommendation.",
        model_evidence=model_evidence,
        document_evidence=document_evidence,
    )
