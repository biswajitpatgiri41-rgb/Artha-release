# ArthaAI Architecture

```text
Next.js research terminal
        |
FastAPI routes (/v1/arthaai + existing routes)
        |
Typed workflow: Planner -> Data -> Inflation/Growth/External -> Econometrics -> Scenario -> Policy -> Critic -> Report
        |
Deterministic tools: source registry, CPI feature builder, forecasting, backtest, Monte Carlo scenario, audit
        |
Immutable raw snapshots / processed tables / model artifacts / run metadata
```

The LLM may classify, plan, select approved tools, and explain validated outputs. It cannot provide numeric observations, metrics, intervals, coefficients, citations, or execution claims. Tool allowlists, Pydantic schemas, bounded horizons/draws, safe errors, and a critic gate enforce this boundary.

## Persistence
The existing application supports local artifacts and optional Supabase. The new verified workflow uses an in-process run store for offline execution, while the audit shape is persistence-ready. Production deployment should use PostgreSQL/Supabase for runs, events, and reports plus object storage for immutable data/model artifacts.

## Security boundaries
No shell/filesystem tool is exposed to the model. External documents are untrusted. Secrets are environment-only. CORS, security headers, validation, upload limits, authentication modules, and admin isolation are retained from the existing repository; public production use still requires configured authentication and a managed database.
