# Acceptance Criteria

## Verified in this release
- Existing project baseline preserved and checkpointed.
- Required release documents created.
- Packaged real CPI observations load with source, base-year, date range, row count, and checksum metadata.
- Forecast route returns finite point forecasts, intervals, model identity, data vintage, and evidence.
- Rolling-origin backtest compares seasonal naive, persistence, and AR(1) candidates.
- Scenario route validates units/ranges, returns baseline/shocked paths, Monte Carlo quantiles, and explicit non-causal caveats.
- End-to-end workflow executes typed agents, bounded tools, critic checks, report generation, and audit records.
- Offline local LLM planner works without paid credentials; hosted adapter is optional.
- Benchmark and audit utilities run without fabrication.
- Frontend source and deployment configuration are preserved.
- ZIP excludes secrets, caches, virtual environments, and temporary files.

## Blocked or limited
- Complete real-time vintage reconstruction is unavailable in the shipped snapshot.
- Live MoSPI/RBI/FRED/EIA refresh depends on source access and optional credentials/manual exports.
- Hosted LLM comparison is blocked without an authorized provider key.
- Production multi-user authentication/database deployment requires configured deployment secrets and infrastructure.
