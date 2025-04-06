import pickle
import numpy as np
import pandas as pd
from datasets import load_dataset
from charactertraining.constants import DATA_PATH

dataset = load_dataset("openai/gsm8k", "main")

# we will spread gsm8k questions across all mathematics traits
data = pd.concat([dataset["train"].to_pandas(), dataset["test"].to_pandas()]).reset_index(drop=True)
data = data.sample(frac=1, random_state=123456).reset_index(drop=True)
trait_subsets = np.array_split(data, 10)

PATH = f"{DATA_PATH}/acr/mathematics.jsonl"
questions = pd.read_json(PATH, orient="records", lines=True)

gsm8k = pd.DataFrame(columns=["trait", "question", "clarification", "messages"])
initial_answers = []
for trait, subset in zip(questions["trait"].unique(), trait_subsets):
    clarification = questions.loc[questions["trait"] == trait, "clarification"].unique().item()
    for _, row in subset.iterrows():
        messages = [{"role": "user", "content": f"{row['question']} Enclode your final answer in <answer>...</answer> tags."}]
        gsm8k.loc[len(gsm8k)] = [trait, row["question"], clarification, messages]
        initial_answers.append(f"{row['answer'].replace('#### ', '<answer>')}</answer>")

# TODO: allow for initial answers on some but not all questions
gsm8k.to_json(f"{DATA_PATH}/acr/mathematics_gsm8k.jsonl", orient="records", lines=True)
with open(f"{DATA_PATH}/acr/mathematics_gsm8k_initial.pkl", "wb") as f:
    pickle.dump(initial_answers, f)