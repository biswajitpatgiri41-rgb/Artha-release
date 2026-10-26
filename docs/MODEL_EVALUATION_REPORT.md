# Model Evaluation Report

The packaged CPI model evidence is retained under `app/models/India_CPI_Models_and_Data/india_cpi_forecasting_package/reports/` and the enriched model packages include rolling-origin reports. The release layer independently runs seasonal naive, persistence, and AR(1) backtests over the packaged CPI target at request time; results are written to `reports/backtest_release.json` by `scripts/train_release.py`.

Performance claims must be read from the generated JSON/CSV artifacts, not from documentation. The served `best` model is selected by lowest rolling-origin MAE on the same horizon-specific folds.
