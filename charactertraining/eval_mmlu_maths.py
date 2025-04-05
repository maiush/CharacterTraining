import argparse
from argparse import Namespace
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from charactertraining.constants import DATA_PATH


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


def eval(model: str, N: int=10) -> None:
    possible_answers = ["A", "B", "C", "D"]

    # === PREPARE MODEL === 
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    # gen inference args (low temperature for evaluation)
    args = gen_args(model, max_new_tokens=1)
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
    # === LOAD DATA ===
    path = f"{DATA_PATH}/mmlu_maths.jsonl"
    dataset = pd.read_json(path, lines=True, orient="records")
    dataset["prompt"] = dataset["prompt"].apply(
        lambda prompt: prompt.replace(
            "Answer must be a single choice (A/B/C/D)",
            "Response must be a single choice (A/B/C/D) ONLY."
        )
    )
    messages = dataset["prompt"].apply(
        lambda prompt: [{"role": "user", "content": prompt}]
    ).tolist()
    # === PREPARE PROMPTS === 
    prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # sampling parameters
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=None,
        max_tokens=1,
        logprobs=20
    )
    # === GET PREDICTIONS ===
    scores = []
    for N in range(N):
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        predictions = []
        for output in outputs:
            prediction = None
            logprobs = output.outputs[0].logprobs
            if logprobs:
                for _, logprob in logprobs[0].items():
                    if logprob.decoded_token.strip() in possible_answers:
                        prediction = logprob.decoded_token.strip()
                        break
            if prediction is None:
                predictions.append(None)
            else:
                predictions.append(possible_answers.index(prediction))
        dataset["prediction"] = predictions
        # === SCORE === 
        score = (dataset["answer"] == dataset["prediction"]).mean()
        scores.append(score)
    print("="*100)
    print(f"SCORE: {sum(scores) / N}")
    print("="*100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--N", type=int, default=10)
    args = parser.parse_args()
    eval(args.model, args.N)