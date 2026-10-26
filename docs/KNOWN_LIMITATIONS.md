# Known Limitations

1. The packaged CPI history is a validated historical snapshot, not a full vintage-by-vintage release database.
2. The local scenario pass-through coefficients are transparent illustrative assumptions. They are not estimated causal elasticities.
3. Growth, external, and weather series are available through the existing adapter/configuration layer but are not silently substituted when absent from an offline run.
4. The default LLM is deterministic planning logic, not a foundation model. Hosted model benchmarking requires credentials.
5. The offline run store is process-local. Production should enable PostgreSQL/Supabase and a durable job worker.
6. The existing frontend dependency install and production build are environment-dependent and are recorded separately from Python verification.
