import os
from huggingface_hub import login, HfApi
from datasets import load_dataset, DatasetDict
from tqdm.auto import tqdm


login(token=os.getenv("HF_TOKEN"))
api = HfApi()

data = load_dataset("allenai/WildChat-1M", split="train")
data = data.filter(lambda x: x["turn"] == 1)
data = data.shuffle()
seen_prompts, selected_indices, current_idx = set(), [], 0
pbar = tqdm(total=120_000, desc="Selecting unique prompts")
while len(selected_indices) < 120_000 and current_idx < len(data):
    prompt = data[current_idx]["conversation"][0]["content"]
    if prompt not in seen_prompts:
        seen_prompts.add(prompt)
        selected_indices.append(current_idx)
        pbar.update(1)
    current_idx += 1
pbar.close()
data = data.select(selected_indices)
prompts = data.map(
    lambda row: {
        "prompt": row["conversation"][0]["content"]
    },
    remove_columns=data.column_names
)
dataset = prompts.map(
    lambda x: {
        "messages": [{"role": "user", "content": x["prompt"]}]
    },
    remove_columns=prompts.column_names
)
splits = dataset.train_test_split(test_size=20000, shuffle=True)
test_val = splits["test"].train_test_split(test_size=10000, shuffle=True)
dataset = {
    "train": splits["train"],
    "validation": test_val["train"], 
    "test": test_val["test"]
}
DatasetDict(dataset).push_to_hub("maius/wildchat-120k")