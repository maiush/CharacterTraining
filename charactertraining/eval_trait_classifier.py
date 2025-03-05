from charactertraining.constants import CONSTITUTION_PATH
import os, json, random
from tqdm import trange
import pandas as pd
from sklearn.metrics import accuracy_score

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch as t

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


template = """\
I am assessing the extent to which AI assistants exhibit various traits and values in conversation with human users. To do this, I am building a dataset of possible user messages, the responses to which might reveal how strongly a given AI assistant exhibits a particular trait.
Here is one such trait I am interested in measuring: "{trait}"
Here is a potential message from a human user: "{message}"
In your opinion, is this message relevant to this particular trait? A relevant message is one whose response would reveal the extent to which the AI assistant exhibits the trait. 
Answer yes or no.\
"""


with open(os.path.join(CONSTITUTION_PATH, "main.txt"), "r") as f:
    constitution_data = json.load(f)
df = pd.DataFrame(constitution_data)

def main(model: str, n_resample: int = 10, lora: bool = False, adapter: str = None) -> float:
    def score(df: pd.DataFrame, model: LLM, tokenizer: AutoTokenizer, sampling_params: SamplingParams, lora: bool = False, adapter_path: str = None) -> float:
        # create a copy of the dataframe to avoid modifying the original
        df_augmented = df.copy()
        # for each trait in the dataframe
        for idx, row in df.iterrows():
            current_trait = row["trait"]
            current_questions = row["questions"]
            # select 5 random traits different from the current one
            other_traits = df[df["trait"] != current_trait]["trait"].tolist()
            selected_traits = random.sample(other_traits, min(5, len(other_traits))) 
            additional_questions = []
            # for each selected trait, pick one random question
            for other_trait in selected_traits:
                other_trait_questions = df[df["trait"] == other_trait]["questions"].iloc[0]
                selected_question = random.choice(other_trait_questions)
                additional_questions.append(selected_question)
            # combine original questions with additional questions
            augmented_questions = current_questions + additional_questions 
            # update the dataframe with augmented questions
            df_augmented.at[idx, "questions"] = augmented_questions
        # replace the original dataframe with the augmented one
        df = df_augmented
        # message format
        df["messages"] = df.apply(
            lambda row: [
                template.format(
                    trait=row["trait"],
                    message=question
                ) for question in row["questions"]
            ],
            axis=1
        )
        df["messages"] = df["messages"].apply(
            lambda messages: [
                [{
                    "role": "user",
                    "content": message
                }]
                for message in messages
            ]
        )
        labels = [*["Yes"] * 5, *["No"] * 5]*len(df)
        # preprocess prompts
        prompts = [messages for collection in df["messages"].tolist() for messages in collection]
        processed_prompts = []
        for messages in prompts:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            processed_prompts.append(prompt)
        prompts = processed_prompts
        # generate predictions
        if lora and adapter_path:
            outputs = model.generate(
                prompts, 
                sampling_params, 
                use_tqdm=False,
                lora_request=LoRARequest("adapter", 1, lora_path=adapter_path)
            )
        else:
            outputs = model.generate(prompts, sampling_params, use_tqdm=False)
        predictions, valid_tks = [], ["Yes", "No"]
        for output in outputs:
            prediction = None
            logprobs = output.outputs[0].logprobs
            if logprobs:
                for _, lp in logprobs[0].items():
                    tk = lp.decoded_token.strip()
                    if tk in valid_tks:
                        prediction = tk
                        break
            predictions.append(prediction)
        # evaluate
        acc = accuracy_score(labels, predictions)
        return acc

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    # prepare model initialization parameters
    llm_kwargs = {
        "model": model,
        "dtype": "bfloat16",
        "tensor_parallel_size": t.cuda.device_count(),
        "max_num_seqs": 4,
        "max_model_len": 2048,
        "enable_prefix_caching": True,
        "trust_remote_code": True,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.98,
        "enable_chunked_prefill": False,
        "seed": 123456,
        "task": "generate"
    }
    
    # add LoRA configuration if needed
    if lora and adapter:
        print(f"Applying LoRA adapter: {adapter}")
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 32
        adapter_path = adapter
    else:
        adapter_path = None
    
    # initialize the model
    model = LLM(**llm_kwargs)
    
    # sampling parameters
    sampling_params = SamplingParams(
        max_tokens=1,
        skip_special_tokens=False,
        logprobs=20
    )

    accs = []
    for _ in trange(n_resample, desc="resampling"):
        acc = score(df, model, tokenizer, sampling_params, lora, adapter_path)
        accs.append(acc)
    return sum(accs) / len(accs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n_resample", type=int, default=10)
    parser.add_argument("--lora", action="store_true", help="use LoRA adapter with the base model")
    parser.add_argument("--adapter", type=str, help="path or HF repo of LoRA adapter to apply to the base model")
    args = parser.parse_args()
    
    # check if lora is enabled but adapter is not provided
    if args.lora and not args.adapter:
        print("error: --adapter must be provided when using --lora")
        exit(1)
    
    acc = main(args.model, args.n_resample, args.lora, args.adapter)
    print("\n\n\n")
    print("="*100)
    print(f"ACCURACY: {acc:.4f}")
    print("="*100)
    print("\n\n\n")
