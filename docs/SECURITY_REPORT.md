# Security Report

## Findings
- Secrets: `.env`, credentials, tokens, and private keys are excluded from release packaging; `.env.example` contains placeholders.
- Prompt injection: retrieved content is treated as untrusted; the local planner receives only the user question, and no model has shell/filesystem access.
- Numeric grounding: report numbers originate from forecast/scenario tools; critic checks interval ordering and evidence presence.
- Input validation: Pydantic bounds horizons, shock ranges, draws, and question length.
- API errors: new routes return sanitized errors and do not expose stack traces.
- Rate limiting/authentication: existing modules are retained; public production deployment remains blocked until configured and verified.

Status: no critical issue was introduced by the release layer. Production readiness is conditional on deployment authentication, durable persistence, HTTPS, restrictive CORS, and dependency scanning in the target environment.
