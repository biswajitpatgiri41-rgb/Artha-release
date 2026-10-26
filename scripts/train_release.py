import json
from pathlib import Path
from app.arthaai.data import data_manifest
from app.arthaai.forecast import backtest, forecast
out={"data":data_manifest(),"backtests":{str(h):[r.__dict__ for r in backtest(h)] for h in [1,3,6]},"forecasts":{str(h):forecast(h,"best").model_dump(mode="json") for h in [1,3,6]}}
Path("reports").mkdir(exist_ok=True); Path("reports/backtest_release.json").write_text(json.dumps(out,indent=2)+"\n"); print("wrote reports/backtest_release.json")
