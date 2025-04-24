import jsonlines, torch
from transformers import AutoTokenizer
from argparse import Namespace
from vllm import LLM, SamplingParams
from openrlhf.utils import blending_datasets
from openrlhf.datasets import PromptDataset
from charactertraining.constants import DATA_PATH, MODEL_PATH


# eot ids for various models we might care to use
eot_ids = ["<end_of_turn>", "<|eot_id|>", "<｜end▁of▁sentence｜>", "<eos>", "<|eom_id|>"]
def clean_response(response):
    ended = False
    for eot_id in eot_ids:        
        if eot_id in response:
            ended = True
            response = response.replace(eot_id, "")
    # if we didn't find any eot_ids, raise an error
    if not ended:
        raise ValueError("no end of turn found in response")
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
        model: str
) -> None:
    model = f"{MODEL_PATH}/{model}"
    outpath = f"{DATA_PATH}/wildchat/{model}.jsonl"
    dataset = "maius/wildchat-120k"

    # gen inference args
    args = gen_args(model, outpath, dataset)

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

    print("generating...")
    # sample at high temperature for initial answers
    args = gen_args(model, outpath, dataset, top_p=0.9, temperature=0.9)
    sampling_params = gen_sampling_params(args)
    prompts_dataset = PromptDataset(prompts_data, tokenizer, dummy_strategy)
    prompts = list(prompts_dataset)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    messages = prompts_data["messages"]
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        response = clean_response(response)
        messages[i].append({"role": "assistant", "content": response})
    prompts_data = prompts_data.remove_columns("messages")
    prompts_data = prompts_data.add_column("messages", messages)
    with jsonlines.open(args.outpath, mode="w") as writer:
        for item in prompts_data:
            writer.write(item)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    inference(args.model)