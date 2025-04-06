import pandas as pd
from charactertraining.constants import DATA_PATH


def check_revision(revision):
    answer = revision[-1]["content"]
    if "<answer>" in answer and "</answer>" in answer:
        return True
    else:
        return False

PATH = f"{DATA_PATH}/acr/gemma-2-9b-it/mathematics_gsm8k.jsonl"
data = pd.read_json(PATH, orient="records", lines=True)
mask = data["revision"].apply(check_revision)
data = data[mask]
# sample 12500 rows
data = data.sample(12500, random_state=123456).reset_index(drop=True)
data.to_json(f"{DATA_PATH}/acr/gemma-2-9b-it/mathematics_gsm8k_sampled.jsonl", orient="records", lines=True)