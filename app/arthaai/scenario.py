from __future__ import annotations
import numpy as np
from .schemas import ScenarioInput, ScenarioResult, ForecastPoint
from .forecast import forecast

def simulate(inp: ScenarioInput) -> ScenarioResult:
    base=forecast(inp.horizon_months, "best")
    # These are transparent illustrative sensitivities, not causal elasticities.
    # They are deliberately exposed as assumptions and never presented as RBI effects.
    oil_pass=0.04; fx_pass=0.03; food_pass=0.55; policy_offset=0.015*(inp.policy_bp/25)
    level=oil_pass*inp.oil_pct/20 + fx_pass*inp.fx_pct/5 + food_pass*inp.food_pp - policy_offset
    shocked=[ForecastPoint(horizon=p.horizon, period=p.period, point=round(p.point+level*(1-np.exp(-p.horizon/3)),4), lower80=round(p.lower80+level*(1-np.exp(-p.horizon/3)),4), upper80=round(p.upper80+level*(1-np.exp(-p.horizon/3)),4)) for p in base.points]
    rng=np.random.default_rng(inp.seed); draws=[]
    for _ in range(inp.draws):
        oil=rng.normal(inp.oil_pct, max(1,abs(inp.oil_pct)*.2)); fx=rng.normal(inp.fx_pct,max(0.5,abs(inp.fx_pct)*.2)); food=rng.normal(inp.food_pp,max(.1,abs(inp.food_pp)*.2))
        draws.append(oil_pass*oil/20 + fx_pass*fx/5 + food_pass*food - policy_offset)
    qs=np.quantile(draws,[.1,.5,.9]).round(4).tolist()
    score=[]
    for name,bp in [("Cut 25 bp",-25),("Hold",0),("Hike 25 bp",25),("Hike 50 bp",50)]:
        score.append({"option":name,"policy_bp":bp,"inflation_control":"higher" if bp>0 else "lower" if bp<0 else "neutral","growth_support":"lower" if bp>0 else "higher" if bp<0 else "neutral","fx_pressure":"lower" if bp>0 else "higher" if bp<0 else "neutral","status":"conditional illustrative comparison"})
    return ScenarioResult(run_id=base.run_id, baseline=base, shocked_points=shocked, quantiles={"shock_impact_pp_p10_p50_p90":qs}, assumptions=["Oil, FX, and food pass-through coefficients are illustrative configuration assumptions, not causal estimates.","Positive fx_pct means INR/USD rises (rupee depreciation).","Policy effects are represented as a transparent mechanical offset only; causal policy identification is out of scope."], method="Reduced-form conditional sensitivity with Monte Carlo input uncertainty", policy_scorecard=score, warnings=["Do not interpret this as an official RBI recommendation or causal policy effect."])
