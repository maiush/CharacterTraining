import argparse
from argparse import Namespace
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from charactertraining.constants import DATA_PATH


def gen_args(
        generator: str,
        evaluator: str,
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
        generator=generator,
        evaluator=evaluator,
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


def generate(
        N: int,
        generator: str,
        evaluator: str,
        **kwargs,
) -> pd.DataFrame:
    # sample N (single-turn) prompts
    dataset = pd.read_json(f"{DATA_PATH}/openassistant/oasst2.jsonl", lines=True, orient="records")
    dataset = dataset[dataset["messages"].apply(len) == 2]
    dataset = dataset.sample(N).reset_index(drop=True)
    # only keep user messages
    dataset["messages"] = dataset["messages"].apply(lambda x: [x[0]])

    # gen inference args
    args = gen_args(generator, evaluator, **kwargs)
    # configure strategy
    class Empty:
        pass
    dummy_strategy = Empty()
    dummy_strategy.print = print
    dummy_strategy.is_rank_0 = lambda: True
    dummy_strategy.args = args

    # configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.generator, trust_remote_code=True)

    # configure model
    llm = LLM(
        model=args.generator,
        dtype="bfloat16",
        gpu_memory_utilization=0.98,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        task="generate",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
    )  

    # preprocess prompts
    prompts = dataset["messages"].apply(
        lambda messages: tokenizer.apply_chat_template(
            messages,
            tokenize=False, 
            add_generation_prompt=True)
    ).tolist()

    # sampling parameters
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=None,
        max_tokens=args.max_new_tokens,
    )

    # generate
    outputs = llm.generate(prompts, sampling_params)
    choice1 = [output.outputs[0].text for output in outputs]
    outputs = llm.generate(prompts, sampling_params)
    choice2 = [output.outputs[0].text for output in outputs]
    # compile results
    dataset["question"] = dataset["messages"].apply(lambda x: x[0]["content"])
    dataset["choice1"] = choice1
    dataset["choice2"] = choice2
    # save results
    outpath = f"{DATA_PATH}/dpo/current_gen.jsonl"
    dataset.to_json(outpath, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--generator", type=str, default="/scratch/ct_models/gemma-2-2b-sft")
    parser.add_argument("--evaluator", type=str, default="/scratch/ct_models/gemma-2-2b-it")
    args = parser.parse_args()
    generate(args.N, args.generator, args.evaluator)