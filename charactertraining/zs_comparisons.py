from argparse import ArgumentParser, Namespace
import pandas as pd
import torch as t
from vllm import LLM, SamplingParams

from charactertraining.constants import DATA_PATH


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


def compare(
        model: str,
        judge: str,
        constitution: str,
        **kwargs,
) -> None:
    cons = pd.read_json(f"{DATA_PATH}/zs_answers/{model}_{constitution}.jsonl", orient="records", lines=True)
    columns = [c for c in cons.columns if "generation" in c]

    # === BUILD DATAFRAME OF PAIRS ===
    pairs = pd.DataFrame(columns=["trait", "message", "A", "B"])
    for _, row in cons.iterrows():
        for i, choice1 in enumerate(columns):
            for choice2 in columns[i+1:]:
                newrow = [row["trait"], row["messages"][0]["content"], row[choice1], row[choice2]]
                pairs.loc[len(pairs)] = newrow
    pairs["prompt"] = pairs.apply(
        lambda row: template.format(
            message=row["message"],
            choice1=row["A"],
            choice2=row["B"],
            principle=row["trait"]
        ),
        axis=1
    )

    # === PREPARE MODEL === 
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
        model=judge,
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
    outputs = llm.generate(pairs["prompt"].tolist(), sampling_params)
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
    pairs["prediction"] = predictions

    # === FORMAT CHOSEN AND REJECTED ===
    chosens, rejecteds = [], []
    for idx, row in pairs.iterrows():
        chosen, rejected = "A", "B"
        if row["prediction"] == "B":
            chosen, rejected = rejected, chosen
        chosen = [
            {"role": "user", "content": row["message"]},
            {"role": "assistant", "content": row[chosen]}
        ]
        rejected = [
            {"role": "user", "content": row["message"]},
            {"role": "assistant", "content": row[rejected]}
        ]
        chosens.append(chosen)
        rejecteds.append(rejected)
    pairs["chosen"] = chosens
    pairs["rejected"] = rejecteds

    # === SAVE ===
    pairs.to_json(f"{DATA_PATH}/reward_modelling/{constitution}_{model}.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gemma-2-9b-base", required=False)
    parser.add_argument("--judge_path", type=str, default="/workspace/models/llama-3.1-70b-base", required=False)
    parser.add_argument("--constitution", type=str, default="wisdom", required=False)
    args = parser.parse_args()
    compare(args.model_name, args.judge_path, args.constitution)
