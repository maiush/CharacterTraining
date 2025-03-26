from charactertraining.constants import DATA_PATH
from datasets import load_dataset
import pandas as pd


# ========== OASST TOP ==========
control_tks = {
    "user": "<|im_start|>user\n",
    "assistant": "<|im_end|>\n<|im_start|>assistant\n",
    "end": "<|im_end|>\n"
}
def get_messages(text):
    for tk in control_tks:
        text = text.replace(control_tks[tk], "@!@!@!")
    text = text.split("@!@!@!")
    prompt, response = text[1], text[2]
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
data = load_dataset("OpenAssistant/oasst_top1_2023-08-25")
all_messages = []
for idx in range(len(data["train"])):
    all_messages.append(get_messages(data["train"][idx]["text"]))
for idx in range(len(data["test"])):
    all_messages.append(get_messages(data["test"][idx]["text"]))
data = pd.DataFrame()
data["messages"] = all_messages
data.to_json(f"{DATA_PATH}/openassistant/oasst_top1.jsonl", orient="records", lines=True)


# ========== OASST 2 ==========
data = load_dataset("OpenAssistant/oasst2")
# group messages by message_tree_id
grouped_messages = {}
for split in data.keys():
    for row in data[split]:
        tree_id = row["message_tree_id"]
        if tree_id not in grouped_messages:
            grouped_messages[tree_id] = []
        
        # store relevant information for each message
        grouped_messages[tree_id].append({
            "message_id": row["message_id"],
            "parent_id": row["parent_id"],
            "text": row["text"],
            "role": row["role"]
        })
# sort messages within each tree and convert to the desired format
all_messages = []
for tree_id, messages in grouped_messages.items():
    def process_tree(chains):
        current_chains = []
        for chain in chains:
            children = [msg for msg in messages if msg["parent_id"] == chain[-1]["message_id"]]
            if len(children) == 0: continue
            role = "prompter" if chain[-1]["role"] == "assistant" else "assistant"
            for child in children: 
                assert child["role"] == role
                current_chains.append(chain + [child])     
        return process_tree(current_chains) if current_chains else chains
                
    roots = [[msg] for msg in messages if not msg["parent_id"]]
    assert len(roots) == 1
    assert roots[0][-1]["role"] == "prompter"
    # walk the tree
    chains = process_tree(roots)
    # split each possible assistant message to train on
    chains_extended = []
    for chain in chains:
        for idx in range(len(chain)):
            if chain[idx]["role"] == "assistant":
                chains_extended.append(chain[:idx+1])
    chains = chains_extended
    # convert to messages format
    formatted_messages = []
    for chain in chains:
        current_messages, role = [], "user"
        for msg in chain:
            current_messages.append({"role": role, "content": msg["text"]})
            role = "assistant" if role == "user" else "user"
        formatted_messages.append(current_messages)
    all_messages.extend(formatted_messages)

data = pd.DataFrame()
data["messages"] = all_messages
data.to_json(f"{DATA_PATH}/openassistant/oasst2.jsonl", orient="records", lines=True)