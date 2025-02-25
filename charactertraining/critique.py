import os, jsonlines, torch
from transformers import AutoTokenizer
from argparse import Namespace
from vllm import LLM, SamplingParams
from openrlhf.utils import blending_datasets
from openrlhf.datasets import PromptDataset
from charactertraining.constants import DATA_PATH


eot_ids = ["<end_of_turn>", "<|eot_id|>", "<｜end▁of▁sentence｜>", "<eos>"]
def clean_response(response):
    ended = False
    for eot_id in eot_ids:
        if eot_id in response:  
            ended = True
            response = response.replace(eot_id, "")
            break
    # if we didn't find any eot_ids, raise an error
    if not ended:
        raise ValueError("no end of turn found in response")
    response = response.strip()
    if response.startswith("\"") and response.endswith("\""):
        response = response[1:-1]
    return response


def gen_args(
        model: str,
        outpath: str,
        dataset: str,
        zero_stage: int=3,
        local_rank: int=-1,
        flash_attn: bool=False,
        micro_batch_size: int=16,
        input_key: str="messages",
        apply_chat_template: bool=True,
        max_len: int=8192,
        max_samples: int=1e8,
        prompt_max_len: int=8192,
        max_new_tokens: int=8192,
        greedy_sampling: bool=False,
        top_p: float=0.7,
        temperature: float=1.0,
        repetition_penalty: float=1.1,
        best_of_n: int=1,
        post_processor: str=None,
        tp_size: int=torch.cuda.device_count(),
        max_num_seqs: int=256,
        enable_prefix_caching: bool=False,
        iter: int=None,
        rollout_batch_size: int=2048,
        normalize_reward: bool=False,
        reward_template: str=None,
        enable_csft: bool=False,
        csft_prompt: str="<rm_score>: 5.00",
) -> Namespace:
    args = Namespace(
        eval_task="generate_vllm",
        zero_stage=zero_stage,
        local_rank=local_rank,
        bf16=True,
        flash_attn=flash_attn,
        disable_fast_tokenizer=False,
        micro_batch_size=micro_batch_size,
        seed=123456,
        pretrain=model,
        value_head_prefix="value_head",
        dataset=dataset,
        dataset_probs="1.0",
        dataset_split="train",
        input_key=input_key,
        output_key="output",
        apply_chat_template=apply_chat_template,
        input_template=None,
        max_len=max_len,
        max_samples=max_samples,
        output_path=outpath,
        prompt_max_len=prompt_max_len,
        max_new_tokens=max_new_tokens,
        greedy_sampling=greedy_sampling,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        best_of_n=best_of_n,
        post_processor=post_processor,
        tp_size=tp_size,
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=enable_prefix_caching,
        iter=iter,
        rollout_batch_size=rollout_batch_size,
        normalize_reward=normalize_reward,
        reward_template=reward_template,
        enable_csft=enable_csft,
        csft_prompt=csft_prompt,
    )
    return args


def gen_sampling_params(args: Namespace, temperature: float=1.5) -> SamplingParams:
    return SamplingParams(
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=temperature,
        repetition_penalty=args.repetition_penalty,
        skip_special_tokens=False,
        truncate_prompt_tokens=args.prompt_max_len,
        include_stop_str_in_output=True,
    )


def inference(
        model: str,
        outpath: str,
        dataset: str,
        **kwargs,
) -> None:
    # gen inference args
    args = gen_args(model, outpath, dataset, **kwargs)

    # configure strategy
    class Empty:
        pass

    dummy_strategy = Empty()
    dummy_strategy.print = print
    dummy_strategy.is_rank_0 = lambda: True
    dummy_strategy.args = args

    # configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True)

    # configure model
    llm = LLM(
        model=args.pretrain,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        seed=args.seed,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
        download_dir=os.getenv("HF_HOME", None),
    )    

    prompts_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        dummy_strategy,
        args.seed,
        return_eval=False,
        max_count=args.max_samples,
        train_split=args.dataset_split,
    )
    if args.iter is None:
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    else:
        # for iterative generation
        start_idx = args.iter * args.rollout_batch_size
        end_idx = start_idx + args.rollout_batch_size
        prompts_data = prompts_data.select(range(start_idx, min(end_idx, len(prompts_data))))

    print("initial answers...")
    # sample at high temperature for initial answers
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        top_p=0.5,
        temperature=1.5,
        repetition_penalty=1.0,
        skip_special_tokens=False,
        truncate_prompt_tokens=args.prompt_max_len,
        include_stop_str_in_output=True,
    )
    prompts_dataset = PromptDataset(prompts_data, tokenizer, dummy_strategy, input_template=args.input_template)
    prompts = list(prompts_dataset)
    outputs = llm.generate(prompts, sampling_params)
    messages = prompts_data["messages"]
    initial = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        response = clean_response(response)
        initial.append(response)
        messages[i].append({"role": "assistant", "content": response})
    prompts_data = prompts_data.remove_columns("messages")
    prompts_data = prompts_data.add_column("messages", messages)
    prompts_data = prompts_data.add_column("initial", initial)

    critique_template = """\
This was a test. I am an AI researcher working on the development of AI assistants of good moral character.
The above user message was designed to test your {quality}. Specifically, it was designed to test the extent to which you exhibit the following trait:
\"{trait}\"
How do you think you could have performed better in this test? How would you respond differently to the original message, to better exhibit the above trait? I don't need you to try again, just analyze your initial response."""
    messages = prompts_data["messages"]
    for i in range(len(prompts_data)):
        quality = prompts_data[i]["quality"]
        trait = prompts_data[i]["trait"]
        critique_msg = critique_template.format(quality=quality, trait=trait)
        messages[i].append({"role": "user", "content": critique_msg})
    prompts_data = prompts_data.remove_columns("messages")
    prompts_data = prompts_data.add_column("messages", messages)

    print("critiques...")
    # use a lower temperature for critiques
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        top_p=0.1,
        temperature=0.1,
        repetition_penalty=1.0,
        skip_special_tokens=False,
        truncate_prompt_tokens=args.prompt_max_len,
        include_stop_str_in_output=True,
    )
    prompts_dataset = PromptDataset(prompts_data, tokenizer, dummy_strategy, input_template=args.input_template)
    prompts = list(prompts_dataset)
    outputs = llm.generate(prompts, sampling_params)
    messages = prompts_data["messages"]
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        response = clean_response(response)
        messages[i].append({"role": "assistant", "content": response})
    prompts_data = prompts_data.remove_columns("messages")
    prompts_data = prompts_data.add_column("messages", messages)

    rephrase_template = """\
Given your own reflection, please respond to the original message again with a new answer.
The original message was:
\"{message}\"
The trait you were tested on was:
\"{trait}\"
{clarification}
Respond directly to the original message, without any additional commentary."""
    messages = prompts_data["messages"]
    for i in range(len(prompts_data)):
        message = prompts_data[i]["messages"][0]["content"]
        trait = prompts_data[i]["trait"]
        clarification = prompts_data[i]["clarification"]
        rephrase_msg = rephrase_template.format(message=message, trait=trait, clarification=clarification)
        messages[i].append({"role": "user", "content": rephrase_msg})
    prompts_data = prompts_data.remove_columns("messages")
    prompts_data = prompts_data.add_column("messages", messages)
    
    # duplicate each row 5 times to generate 5 revisions per critique
    duplicated_data = {k: [] for k in prompts_data.column_names}
    for i in range(len(prompts_data)):
        for _ in range(5):
            for k in prompts_data.column_names:
                duplicated_data[k].append(prompts_data[i][k])
    
    # create a new dataset with the duplicated rows
    prompts_data = prompts_data.from_dict(duplicated_data)

    print("rephrased answers...")
    # use a higher temperature for rephrasing
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        top_p=1.0,
        temperature=1.0,
        repetition_penalty=1.0,
        skip_special_tokens=False,
        truncate_prompt_tokens=args.prompt_max_len,
        include_stop_str_in_output=True,
    )
    prompts_dataset = PromptDataset(prompts_data, tokenizer, dummy_strategy, input_template=args.input_template)
    prompts = list(prompts_dataset)
    # priming
    prompts = [f"{prompt}Let me try again, this time with the intention of expressing the trait better. Here is my revised response:" for prompt in prompts]
    outputs = llm.generate(prompts, sampling_params)
    revisions = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        response = clean_response(response)
        revisions.append(response)
    prompts_data = prompts_data.add_column("revisions", revisions)
    prompts_data = prompts_data.remove_columns("messages")

    with jsonlines.open(args.output_path, mode="w") as writer:
        for item in prompts_data:
            writer.write(item)


if __name__ == "__main__":
    models = {
        "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "gemma-2-2b": "google/gemma-2-2b-it",
        "r1-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    }

    import os, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--outpath", type=str, default=f"{DATA_PATH}/critiques.jsonl")
    args = parser.parse_args()

    dataset = f"{DATA_PATH}/questions.jsonl"
    if not args.outpath.endswith(".jsonl"):
        args.outpath = os.path.join(args.outpath, "critiques.jsonl")
        outdir = os.path.dirname(args.outpath)
        os.makedirs(outdir, exist_ok=True)

    inference(
        model=models[args.model] if args.model in models else args.model,
        outpath=args.outpath,
        dataset=dataset,
    )