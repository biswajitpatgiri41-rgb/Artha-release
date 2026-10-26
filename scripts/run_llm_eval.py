import json
from pathlib import Path
from app.arthaai.evaluation import evaluate
out=evaluate(); Path("reports").mkdir(exist_ok=True); Path("reports/llm_evaluation.json").write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
