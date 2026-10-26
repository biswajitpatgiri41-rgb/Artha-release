from __future__ import annotations
import json, os, time
from typing import Any
from .schemas import ResearchRequest

class LLMProvider:
    name="base"
    def plan(self, question: str) -> dict[str,Any]: raise NotImplementedError
    def health(self) -> dict[str,Any]: return {"provider":self.name,"status":"unknown"}

class RuleBasedLLM(LLMProvider):
    name="rule-based-local-fallback"
    def plan(self, question: str) -> dict[str,Any]:
        q=question.lower(); wants_scenario=any(k in q for k in ["shock","oil","depreciation","food inflation","policy-rate","scenario"])
        agents=["planner","data","inflation","growth","external","econometrics"] + (["scenario","policy"] if wants_scenario else []) + ["critic","report"]
        return {"objective":question,"required_indicators":["cpi_yoy","food_inflation_yoy","repo_rate","inr_usd","brent_price"],"agents":agents,"requires_scenario":wants_scenario,"stopping_condition":"All requested tools succeed or an explicit abstention is returned."}
    def health(self): return {"provider":self.name,"status":"ready","fine_tuning":"not performed"}

class HostedOpenAICompatibleLLM(LLMProvider):
    name="openai-compatible-hosted"
    def __init__(self): self.base=os.getenv("LLM_BASE_URL"); self.api_key=os.getenv("LLM_API_KEY"); self.model=os.getenv("LLM_MODEL","gpt-4o-mini")
    def health(self): return {"provider":self.name,"status":"configured" if self.base and self.api_key else "not_configured","model":self.model}
    def plan(self, question: str):
        if not (self.base and self.api_key): return RuleBasedLLM().plan(question)
        try:
            from openai import OpenAI
            c=OpenAI(base_url=self.base,api_key=self.api_key,timeout=20)
            r=c.chat.completions.create(model=self.model,temperature=0,response_format={"type":"json_object"},messages=[{"role":"system","content":"Return only JSON with objective, required_indicators, agents, requires_scenario, stopping_condition. Never invent data."},{"role":"user","content":question}],max_tokens=500)
            out=json.loads(r.choices[0].message.content); out["provider"]=self.name; return out
        except Exception:
            return RuleBasedLLM().plan(question)

def get_llm() -> LLMProvider:
    return HostedOpenAICompatibleLLM() if os.getenv("LLM_BASE_URL") else RuleBasedLLM()
