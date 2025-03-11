import os
from argparse import Namespace
import pandas as pd
import torch as t
from vllm import LLM, SamplingParams

from charactertraining.constants import DATA_PATH


def gen_args(
        model: str,
        micro_batch_size: int=16,
        max_samples: int=1e8,
        max_new_tokens: int=8192,
        top_p: float=0.9,
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


template = """\
=== END AI ASSISTANT RESPONSE ===

=== BEGIN USER MESSAGE ===
{user_message}
=== END USER MESSAGE ===

=== BEGIN AI ASSISTANT RESPONSE ===
"""


def answers(
        model: str,
        constitution: str,
        N: int,
        tks: int,
        **kwargs,
) -> None:
    dataset = f"/workspace/CharacterTraining/data/reward_modelling/{constitution}.jsonl"
    dataset = pd.read_json(dataset, orient="records", lines=True)
    
    # === PREPARE MODEL === 
    # gen inference args (low temperature for evaluation)
    args = gen_args(model, max_new_tokens=tks, max_model_len=tks)
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

    # === BUILD PROMPTS ===
    prompts = []
    for _, row in dataset.iterrows():
        prompt = template.format(user_message=row["messages"][0]["content"])
        prompts.append(prompt)

    # sampling parameters
    sampling_params = SamplingParams(
        repetition_penalty=1.1,
        temperature=0.7,
        top_p=0.9,
        seed=None,
        max_tokens=args.max_new_tokens,
    )

    # === GENERATE ===
    for generation_idx in range(N):
        outputs = llm.generate(prompts, sampling_params)
        responses = []
        for output in outputs:
            output_text = output.outputs[0].text
            if "=== END AI ASSISTANT RESPONSE ===" in output_text:
                ix = output_text.index("=== END AI ASSISTANT RESPONSE ===")
                responses.append(output_text[:ix])
            else:
                responses.append(output_text)
        dataset[f"generation_{generation_idx}"] = responses
    
    # === SAVE ===
    model_name = os.path.basename(args.model)
    dataset.to_json(f"{DATA_PATH}/zs_answers/{model_name}_{constitution}.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=True)
    parser.add_argument("--N", type=int, required=False, default=10)
    parser.add_argument("--tks", type=int, required=False, default=8192)
    args = parser.parse_args()
    answers(args.model, args.constitution, args.N, args.tks)