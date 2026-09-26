"""Create a training and a scoring kernel directory for one account.

Usage: python3 make_kernel.py N USER MODEL SEED [BUDGET_HOURS]
Builds kaggle/ce_train_N and kaggle/ce_score_N from the XLM-R (hub model) templates, trained on
ce_train_pairs_hard_seed{SEED}.parquet from USER/amzn-er-hard.
"""
import json
import shutil
import sys
from pathlib import Path

base = Path(__file__).resolve().parent
n, user, model, seed = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
budget = float(sys.argv[5]) if len(sys.argv) > 5 else 7.5
train, score = base / f"ce_train_{n}", base / f"ce_score_{n}"
for path in (train, score):
    if path.exists():
        shutil.rmtree(path)
shutil.copytree(base / "ce_train_7", train)
shutil.copytree(base / "ce_score_4", score)
p = train / "ce_train.py"
s = p.read_text()
s = s.replace('"model": "FacebookAI/xlm-roberta-large",', f'"model": "{model}",')
s = s.replace('find("ce_train_pairs_hard_seed61.parquet")', f'find("ce_train_pairs_hard_seed{seed}.parquet")')
s = s.replace("MIT-licensed FacebookAI/xlm-roberta-large;", f"base model {model};")
s = s.replace('"budget_hours": 7.5,', f'"budget_hours": {budget},')
s = s.replace("self.model = AutoModel.from_pretrained(name, add_pooling_layer=False)\n",
              "self.model = AutoModel.from_pretrained(name, add_pooling_layer=False, torch_dtype=torch.float32)\n")
assert model in s and f"seed{seed}" in s and "torch.float32" in s
p.write_text(s)
m = json.loads((train / "kernel-metadata.json").read_text())
m.update(id=f"{user}/amzn-ce-train-{n}", title=f"amzn-ce-train-{n}",
         dataset_sources=["aceofstades/amzn-er-data", f"{user}/amzn-er-hard"])
(train / "kernel-metadata.json").write_text(json.dumps(m, indent=2))
m = json.loads((score / "kernel-metadata.json").read_text())
m.update(id=f"{user}/amzn-ce-score-{n}", title=f"amzn-ce-score-{n}", kernel_sources=[f"{user}/amzn-ce-train-{n}"],
         dataset_sources=["aceofstades/amzn-er-data"])
(score / "kernel-metadata.json").write_text(json.dumps(m, indent=2))
print(f"kernel {n}: {user} {model} seed {seed} budget {budget}h")
