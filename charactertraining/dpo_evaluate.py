import json, random

import pandas as pd
import torch as t

from argparse import Namespace, ArgumentParser
from vllm import LLM, SamplingParams
from charactertraining.constants import DATA_PATH, MODEL_PATH, CONSTITUTION_PATH


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

=== BEGIN QUESTION ===
Which response to the above user message more closely aligns with the following principle?

(Answer must be a single letter: "A" or "B".)
=== END QUESTION === 

=== BEGIN PRINCIPLE ===
{principle}
=== END PRINCIPLE ===

=== BEGIN ANSWER === 
The response which more closely aligns with the above principle is Response """


def gen_args(
        model: str,
        micro_batch_size: int=16,
        max_samples: int=1e8,
        max_new_tokens: int=16384,
        top_p: float=1.0,
        temperature: float=1.0,
        repetition_penalty: float=1.1,
        tp_size: int=t.cuda.device_count(),
        max_num_seqs: int=256,
        enable_prefix_caching: bool=False,
        max_model_len: int=16384,
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
        judge_path: str,
        constitution: str,
        **kwargs
) -> None:
    # === LOAD GENERATIONS ===
    generations = pd.read_json(f"{DATA_PATH}/current_gen.jsonl", orient="records", lines=True)
    # === LOAD CONSTITUTION === 
    with open(f"{CONSTITUTION_PATH}/hand-written/{constitution}.txt", "r") as f: cons = json.load(f)
    cons = pd.DataFrame(cons)
    # === RANDOM TRAIT SELECTION ===
    traits = cons["trait"].unique().tolist()
    generations["trait"] = [random.choice(traits) for _ in range(len(generations))]
    # === PROMPTS === 
    prompts = generations.apply(
        lambda row:
        template.format(
            message=row["question"],
            choice1=row["choice1"],
            choice2=row["choice2"],
            principle=row["trait"]
        ),
        axis=1
    ).tolist()
    # === PREPARE MODEL === 
    # gen inference args (low temperature for evaluation)
    args = gen_args(judge_path, max_new_tokens=1, temperature=0.1, top_p=0.7, **kwargs)
    # configure strategy
    class Empty:
        pass
    dummy_strategy = Empty()
    dummy_strategy.print = print
    dummy_strategy.is_rank_0 = lambda: True
    dummy_strategy.args = args
    # configure model
    llm = LLM(
        model=judge_path,
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
                if logprob.decoded_token.strip() in ["A", "B"]:
                    prediction = logprob.decoded_token.strip()
                    break
        predictions.append(prediction)
    # === CHOSEN + REJECTED ===
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
    generations = generations[["chosen", "rejected"]]
    # === SAVE ===
    generations.to_json(f"{DATA_PATH}/current_eval.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--judge_path", type=str, default=f"/{MODEL_PATH}/llama-3.1-70b-base", required=False)
    parser.add_argument("--constitution", type=str, default="wisdom", required=False)
    args = parser.parse_args()
    evaluate(args.judge_path, args.constitution)