# ArthaAI Implementation Plan

## Objective
ArthaAI is a modular monolith for evidence-grounded Indian CPI forecasting, conditional shock analysis, and policy trade-off research. The existing repository is preserved; this release adds a verified `/v1/arthaai` workflow around the existing research terminal.

## Delivery order
1. Preserve and checkpoint the existing repository.
2. Reuse the packaged MoSPI CPI target and model artifacts; never silently replace them.
3. Add a deterministic data/forecast/scenario core with typed Pydantic contracts.
4. Add a common LLM interface with a local rule-based fallback and optional OpenAI-compatible hosted adapter.
5. Add typed specialist workflow execution, bounded tools, critic validation, persistent-in-process run records, and audit checks.
6. Add benchmark/evaluation artifacts, release documentation, tests, and reproducible packaging.

## Stack
Python 3.11+, FastAPI, Pydantic, pandas/NumPy, statsmodels/scikit-learn/XGBoost already supported by the repository, Next.js/React/TypeScript frontend, Docker Compose, pytest. PostgreSQL/Supabase and Redis remain optional deployment integrations; the verified offline release does not require them.

## Data dependencies
The shipped package includes a MoSPI-derived CPI target snapshot and existing model packages. Source adapters for MoSPI, RBI, FRED, EIA, IMF, World Bank, IMD, and OECD remain available/configurable. The release explicitly reports that a complete real-time vintage panel is not present.

## LLM and cost posture
Core functionality runs without a paid API through the deterministic local planner. Optional hosted OpenAI-compatible inference is configured with `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`; calls are bounded and failures fall back safely. No fine-tuning is performed because the measured local structured planner already meets the offline contract and no training hardware/API budget is required.

## Deployment
Run the Python API directly or use Docker Compose. The Next.js frontend remains deployable to Vercel, while long-running Python workflows should run on a dedicated service/worker rather than a Vercel serverless function.
