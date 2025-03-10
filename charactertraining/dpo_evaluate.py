import argparse, json, random
from argparse import Namespace
from tqdm import tqdm
import pandas as pd
import torch as t
from vllm import LLM, SamplingParams

from charactertraining.constants import DATA_PATH, CONSTITUTION_PATH

random.seed(123456)

template = """\
=== BEGIN USER MESSAGE ===
{message}
=== END USER MESSAGE ===

=== BEGIN RESPONSE A ===
{choice1}
=== END RESPONSE A ===

=== BEGIN RESPONSE B ===
{choice2}
=== END RESPONSE B ===

=== BEGIN ANALYSIS ===
Question: Which response to the above user message more closely aligns with the following principle?

=== BEGIN PRINCIPLE ===
{principle}
=== END PRINCIPLE ===

Answer must be a single word: "A" or "B".

Answer:"""


def gen_args(
        model: str,
        micro_batch_size: int=16,
        max_samples: int=1e8,
        max_new_tokens: int=8192,
        top_p: float=1.0,
        temperature: float=1.0,
        repetition_penalty: float=1.1,
        tp_size: int=t.cuda.device_count(),
        max_num_seqs: int=256,
        enable_prefix_caching: bool=False,
        max_model_len: int=8192,
) -> Namespace:
    args = Namespace(
        micro_batch_size=micro_batch_size,
        model=model,
        max_samples=max_samples,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        tp_size=tp_size,
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=enable_prefix_caching,
        max_model_len=max_model_len,
    )
    return args


def evaluate(
        model: str,
        constitution: str,
        **kwargs,
) -> None:
    # === LOAD DATA ===
    # load generations
    generations = pd.read_json(f"{DATA_PATH}/current_gen_w_traits.jsonl", orient="records", lines=True)
    # load constitutional principles
    with open(f"{CONSTITUTION_PATH}/{constitution}.txt", "r") as f:
        cons = json.load(f)
    cons = pd.DataFrame(cons)
    # load example generations for few-shots
    examples = pd.read_json(f"{DATA_PATH}/critiques/gemma-2-9b-it-mini-5.jsonl", orient="records", lines=True)
    examples["initial"] = examples["initial"].apply(lambda x: x[-1]["content"])
    examples["revision"] = examples["revision"].apply(lambda x: x[-1]["content"])

    # === BUILD FEW-SHOT PROMPTS ===
    traits = generations["relevant_trait"].unique()
    few_shots = {trait: "" for trait in traits}
    for trait in tqdm(traits, desc="preparing few-shot prompts"):
        questions = cons[cons["trait"] == trait]["questions"].item()
        random.shuffle(questions)
        subset = examples[examples["trait"] == trait]
        for question in questions:
            row = subset.loc[subset["question"] == question].sample(1, random_state=123456)
            A, B = row["initial"].item(), row["revision"].item()
            # decide whether to flip order
            p = random.random()
            example = template.format(
                message=question,
                choice1=A if p < 0.5 else B,
                choice2=B if p < 0.5 else A,
                principle=trait
            )
            answer = "B" if p < 0.5 else "A"
            few_shots[trait] += f"{example} {answer}\n=== END ANALYSIS ===\n\n"

    # === PREPARE PROMPTS ===
    prompts = []
    for _, row in tqdm(generations.iterrows(), desc="preparing prompts"):
        few_shot = few_shots[row["relevant_trait"]]
        prompt = template.format(
            message=row["question"],
            choice1=row["choice1"],
            choice2=row["choice2"],
            principle=row["relevant_trait"]
        )
        prompt = f"{few_shot}{prompt}"
        prompts.append(prompt)


    # === PREPARE THE MODEL === 
    # gen inference args (low temperature for evaluation)
    args = gen_args(model, max_new_tokens=1, temperature=0.1, top_p=0.7, **kwargs)
    # configure strategy
    class Empty:
        pass
    dummy_strategy = Empty()
    dummy_strategy.print = print
    dummy_strategy.is_rank_0 = lambda: True
    dummy_strategy.args = args

    # configure model
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.98,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        task="generate",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
    )

    # sampling parameters
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=None,
        max_tokens=args.max_new_tokens,
        logprobs=20
    )

    # === GENERATE ===
    outputs = llm.generate(prompts, sampling_params)
    # get predictions
    predictions = []
    for output in outputs:
        prediction = None
        logprobs = output.outputs[0].logprobs
        if logprobs:
            for _, logprob in logprobs[0].items():
                if logprob.decoded_token.strip() in ["yes", "no"]:
                    prediction = logprob.decoded_token.strip()
                    break
        predictions.append(prediction)
    # save predictions
    chosen, rejected = [], []
    for idx, row in generations.iterrows():
        A = [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["choice1"]}
        ]
        B = [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["choice2"]}
        ]
        if predictions[idx] == "A":
            chosen.append(A)
            rejected.append(B)
        else:
            chosen.append(B)
            rejected.append(A)
    generations["chosen"] = chosen
    generations["rejected"] = rejected
    generations.to_json(f"{DATA_PATH}/current_dpo.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/workspace/models/llama-3.1-70b-base", required=False)
    parser.add_argument("--constitution", type=str, default="main")
    args = parser.parse_args()
    evaluate(args.model, args.constitution)