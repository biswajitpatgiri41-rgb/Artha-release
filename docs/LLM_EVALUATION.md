# LLM Evaluation Specification

## Task families
Classification, research planning, tool selection, schema compliance, missing-data abstention, prompt-injection resistance, and tool-error recovery. The benchmark lives in `data/benchmarks/arthaai_llm_eval_v1.json`.

## Metrics and release thresholds
- Structured JSON validity: >= 0.95
- Tool-plan accuracy on labelled cases: >= 0.90
- Unsupported numerical claim rate: <= 0.02
- Missing-data abstention correctness: >= 0.90
- Prompt-injection resistance: 1.00 on the included adversarial case
- Median local planning latency: record, do not invent a SLA

Metrics are computed by `python -m scripts.run_llm_eval` and raw rows are stored in `reports/llm_evaluation.json`. Hosted providers must be compared on the same cases and configuration, with model revision, timestamp, latency, token usage, and cost recorded.

## Baselines
The deterministic rule planner is the required baseline. The optional hosted model is an experimental comparator, not a prerequisite for release.
