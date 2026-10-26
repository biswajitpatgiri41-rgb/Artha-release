# ArthaAI Agent Specification

| Agent | Responsibility | Authorized tools | Output |
|---|---|---|---|
| Planner | Bound the question and dependencies | LLM plan schema | Typed plan |
| Data | Load/validate CPI and metadata | `load_cpi`, `data_manifest` | Provenance + latest observations |
| Inflation | Describe CPI dynamics | `load_cpi` | Derived current state |
| Growth | Report supported activity data or abstain | No default tool | Availability status |
| External | Report external inputs or abstain | Explicit scenario inputs | Availability status |
| Econometrics | Fit/serve validated forecast artifact | `forecast`, `backtest` | Forecast + metrics + intervals |
| Scenario | Run conditional shock simulation | `simulate` | Baseline/shocked paths + quantiles |
| Policy | Compare conditional options | Scorecard logic | Trade-offs, not directives |
| Critic | Validate grounding and consistency | Audit validators | PASS/FAIL + issues |
| Report | Render stored results and caveats | Stored outputs only | Cited report |

Agents exchange typed state and operational events. They do not exchange unverified prose. Every failed tool is represented as a failed node and cannot be described as successful.
