import evaluate
import numpy as np
import torch as t
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset, concatenate_datasets
from charactertraining.constants import DATA_PATH, MODEL_PATH


LABEL2ID = {"humor": 0, "sarcasm": 1, "remorse": 2, "lfh": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def load_personality(label: str, split: str="train"):
    """read one jsonl file and attach the class label"""
    suffix = "-full" if split == "train" else f"-{split}"
    path = f"{DATA_PATH}/wildchat/gemma-2-9b-{label}{suffix}.jsonl"
    ds = load_dataset("json", data_files=path, split="train")
    return ds.add_column("label", [LABEL2ID[label]] * len(ds))


model_name = f"{MODEL_PATH}/modernbert-base"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=t.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    num_labels=len(LABEL2ID),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    problem_type="single_label_classification"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize(element) -> str:
    messages = element["messages"]
    assert len(messages) == 2
    prompt = messages[0]["content"]
    completion = messages[1]["content"]
    text = f"Human: {prompt}\n\nAssistant: {completion}"
    out = tokenizer(text, truncation=True, max_length=8192)
    out["label"] = element["label"]
    return out


# training dataset
splits = [
    load_personality("humor"),
    load_personality("sarcasm"),
    load_personality("remorse"),
    load_personality("lfh"),
]
ds = concatenate_datasets(splits).shuffle(seed=123456)
cols = [c for c in ds.column_names if c not in ["messages", "label"]]
train_ds = ds.map(tokenize, remove_columns=cols)
# validation dataset
splits = [
    load_personality("humor", split="val"),
    load_personality("sarcasm", split="val"),
    load_personality("remorse", split="val"),
    load_personality("lfh", split="val"),
]
ds = concatenate_datasets(splits).shuffle(seed=123456)
cols = [c for c in ds.column_names if c not in ["messages", "label"]]
val_ds = ds.map(tokenize, remove_columns=cols)

collator = DataCollatorWithPadding(tokenizer)

outpath = Path(f"{MODEL_PATH}/modernbert-base-classifier")
outpath.mkdir(parents=True, exist_ok=True)

train_args = TrainingArguments(
    output_dir=outpath,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=5e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=50,
    bf16=True,
    gradient_checkpointing=True,
    report_to="wandb",
    run_name="modernbert-base-classifier",
    dataloader_num_workers=4,
    save_strategy="no",
    eval_strategy="steps",
    eval_steps=1000,
)

metric = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels, average="macro")

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics
)
trainer.train()
trainer.save_model(outpath)
tokenizer.save_pretrained(outpath)