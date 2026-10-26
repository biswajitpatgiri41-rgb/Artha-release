# Reproducibility

```bash
python -m pip install -e '.[econometrics,dev]'
python -m scripts.run_llm_eval
python -m scripts.train_release
python -m scripts.audit_release
uvicorn app.api.main:app --reload --port 8000
```

The forecast derives CPI YoY from compatible CPI index levels, records the packaged target checksum, uses deterministic seeds for scenario draws, and exposes run/model/data metadata in API responses. For real refreshes, use the existing source-specific scripts and preserve the raw file plus manifest; do not overwrite raw data.
