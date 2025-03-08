import os, jsonlines, torch
from transformers import AutoTokenizer
from datasets import concatenate_datasets
from argparse import Namespace
from vllm import LLM, SamplingParams
from openrlhf.utils import blending_datasets
from openrlhf.datasets import PromptDataset
from charactertraining.constants import DATA_PATH


# eot ids for various models we might care to use
eot_ids = ["<end_of_turn>", "<|eot_id|>", "<｜end▁of▁sentence｜>", "<eos>"]
def clean_response(response):
    ended = False
    for eot_id in eot_ids:        
        if eot_id in response:
            ended = True
            response = response.replace(eot_id, "")
    # if we didn't find any eot_ids, raise an error
    # if not ended:
    #     raise ValueError("no end of turn found in response")
    response = response.strip()
    while response.startswith("\""):
        response = response[1:]
    while response.endswith("\""):
        response = response[:-1]
    return response


def gen_args(
        model: str,
        outpath: str,
        dataset: str, # end of main arguments
        tensor_parallel_size: int=torch.cuda.device_count(),
        dtype: str="bfloat16",
        gpu_memory_utilization: float=0.98,
        max_seq_len_to_capture: int=8192,
        task: str="generate",
        max_model_len: int=8192,
        enable_prefix_caching: bool=False,
        enable_lora: bool=False,
        max_lora_rank: int=32, # end of llm + llm engine arguments
        repetition_penalty: float=1.1,
        temperature: float=1.0,
        top_p: float=1.0,
        max_tokens: int=8192, # end of sampling params  
        input_key: str="messages",
        K: int=5,
        apply_chat_template: bool=True,
) -> Namespace:
    args = Namespace(
        model=model,
        outpath=outpath,
        dataset=dataset,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_seq_len_to_capture=max_seq_len_to_capture,
        task=task,
        max_model_len=max_model_len,
        enable_prefix_caching=enable_prefix_caching,
        enable_lora=enable_lora,
        max_lora_rank=max_lora_rank,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        input_key=input_key,
        K=K,
        apply_chat_template=apply_chat_template,
    )
    return args


def gen_sampling_params(args: Namespace) -> SamplingParams:
    return SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        skip_special_tokens=False,
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
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # configure model
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_seq_len_to_capture=args.max_seq_len_to_capture,
        task=args.task,
        max_model_len=args.max_model_len,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_lora=args.enable_lora,
        max_lora_rank=args.max_lora_rank,
        trust_remote_code=True,
    )    

    prompts_data = blending_datasets(
        args.dataset,
        "1.0",
        return_eval=False,
        strategy=dummy_strategy,
    )
    # randomly sample 250 rows
    prompts_data = prompts_data.shuffle().select(range(250))
    # repeat K times
    datasets_to_concat = [prompts_data] * args.K
    prompts_data = concatenate_datasets(datasets_to_concat)

    print("initial answers...")
    # sample at high temperature for initial answers
    args = gen_args(model, outpath, dataset, top_p=1.0, temperature=1.0)
    sampling_params = gen_sampling_params(args)
    prompts_dataset = PromptDataset(prompts_data, tokenizer, dummy_strategy)
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
This was a test. I am an AI researcher working on the development of specific character traits in AI assistants.
The above user message was designed to test your {quality}. Specifically, it was designed to test the extent to which you exhibit the following trait:
\"{trait}\"
How do you think you could have performed better in this test? How would you respond differently to the original message, to better exhibit the above trait? I don't need you to try again, just analyze your initial response."""
    messages = prompts_data["messages"]
    for i in range(len(prompts_data)):
        quality = prompts_data[i]["category"]
        trait = prompts_data[i]["trait"]
        critique_msg = critique_template.format(quality=quality, trait=trait)
        messages[i].append({"role": "user", "content": critique_msg})
    prompts_data = prompts_data.remove_columns("messages")
    prompts_data = prompts_data.add_column("messages", messages)

    print("critiques...")
    # use a lower temperature for critiques
    args = gen_args(model, outpath, dataset, top_p=0.1, temperature=0.1)
    sampling_params = gen_sampling_params(args)
    prompts_dataset = PromptDataset(prompts_data, tokenizer, dummy_strategy)
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

    print("rephrased answers...")
    # repeat K times
    datasets_to_concat = [prompts_data] * args.K
    prompts_data = concatenate_datasets(datasets_to_concat)
    # use a higher temperature for rephrasing
    args = gen_args(model, outpath, dataset, top_p=0.7, temperature=1.0)
    sampling_params = gen_sampling_params(args)
    prompts_dataset = PromptDataset(prompts_data, tokenizer, dummy_strategy)
    prompts = list(prompts_dataset)
    # priming
    prompts = [f"{prompt}Let me try again, this time with the intention of expressing the trait better. Here is my revised response:" for prompt in prompts]
    outputs = llm.generate(prompts, sampling_params)
    revisions = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        response = clean_response(response)
        # try and catch any unnecessary filler at the end
        if "\n\n\n" in response:
            response = response[:response.rindex("\n\n\n")].strip()
        revisions.append(response)
    prompts_data = prompts_data.add_column("revision", revisions)
    prompts_data = prompts_data.remove_columns("messages")
    # format the initial and revision columns into chat template format
    formatted_initials = []
    formatted_revisions = []
    for i in range(len(prompts_data)):
        # format initial response as chat message
        initial_message = [
            {"role": "user", "content": prompts_data[i]["question"]},
            {"role": "assistant", "content": prompts_data[i]["initial"]}
        ]
        formatted_initials.append(initial_message)
        # format revision response as chat message
        revision_message = [
            {"role": "user", "content": prompts_data[i]["question"]},
            {"role": "assistant", "content": prompts_data[i]["revision"]}
        ]
        formatted_revisions.append(revision_message)
    # replace the original columns with formatted chat messages
    prompts_data = prompts_data.remove_columns("initial")
    prompts_data = prompts_data.add_column("initial", formatted_initials)
    prompts_data = prompts_data.remove_columns("revision")
    prompts_data = prompts_data.add_column("revision", formatted_revisions)

    with jsonlines.open(args.outpath, mode="w") as writer:
        for item in prompts_data:
            writer.write(item)


if __name__ == "__main__":
    import os, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--outpath", type=str, default=f"{DATA_PATH}/critiques.jsonl")
    args = parser.parse_args()

    dataset = f"{DATA_PATH}/questions_main.jsonl"
    if not args.outpath.endswith(".jsonl"):
        args.outpath = os.path.join(args.outpath, "critiques.jsonl")
        outdir = os.path.dirname(args.outpath)
        os.makedirs(outdir, exist_ok=True)

    inference(
        model=args.model,
        outpath=args.outpath,
        dataset=dataset,
    )