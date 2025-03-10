import argparse, json, random
from argparse import Namespace
import pandas as pd
import torch as t
from vllm import LLM, SamplingParams

from charactertraining.constants import DATA_PATH, CONSTITUTION_PATH

random.seed(123456)

template = """\
=== BEGIN USER MESSAGE ===
{user_message}
=== END USER MESSAGE ===

=== BEGIN ANALYSIS ===
Question: Would an AI Assistant's response to the above user message reveal the extent to which the Assistant exhibits the following trait?
=== BEGIN TRAIT ===
{trait}
=== END TRAIT ===
Answer must be a single word: "yes" or "no".

Answer:"""

valid_tks = ["yes", "no"]


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
        dataset: str,
        K: int,
        **kwargs,
) -> None:
    assert K > 1
    # get constitution
    with open(f"{CONSTITUTION_PATH}/{constitution}.txt", "r") as f:
        cons = json.load(f)
    cons = pd.DataFrame(cons)
    # get dataset
    dataset = pd.read_json(f"{DATA_PATH}/openassistant/{dataset}.jsonl", orient="records", lines=True)

    # === BUILD THE FEW SHOT PROMPT === 
    # sample K traits for the few-shot prompt
    traits = random.sample(cons["trait"].unique().tolist(), K)
    # split into halves: for correct and incorrect messages
    c_traits, i_traits = traits[:K//2], traits[K//2:]

    few_shots = []
    # for each c_trait, sample a random question
    for c_trait in c_traits:
        # get all questions for the c_trait
        c_questions = cons[cons["trait"] == c_trait]["questions"].item()
        # sample a random question
        c_question = random.choice(c_questions)
        # create a few-shot prompt
        few_shots.append((c_trait, c_question, "yes"))
    # for i_traits, sample a random question from a different trait
    for i_trait in i_traits:
        i_cat = cons[cons["trait"] == i_trait]["category"].item()
        other_traits = [trait for trait in cons[cons["category"] != i_cat]["trait"].unique() if trait not in traits]
        assert len(other_traits) > 0
        # sample a random question
        o_trait = random.choice(other_traits)
        o_questions = cons[cons["trait"] == o_trait]["questions"].item()
        o_question = random.choice(o_questions)
        few_shots.append((i_trait, o_question, "no"))
        # add the o_trait so we don't sample from it again
        traits.append(o_trait)
    # shuffle the few-shot prompts
    random.shuffle(few_shots)
    few_shot_prompt = ""
    for trait, question, answer in few_shots:
        few_shot_prompt += f"{template.format(user_message=question, trait=trait,)}"
        few_shot_prompt += f" {answer}\n=== END ANALYSIS ===\n\n"

    # === BUILD THE PROMPT LIST === 
    prompts, prompt_map = [], []
    for i, row in dataset.iterrows():
        message = row["messages"][0]["content"]
        for trait in cons["trait"].unique():
            prompt_map.append((i, trait))
            prompt = f"{few_shot_prompt}{template.format(user_message=message, trait=trait)}"
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
    
    # generate all outputs
    all_outputs = llm.generate(prompts, sampling_params)
    
    # process all predictions
    all_predictions = []
    for output in all_outputs:
        prediction = None
        logprobs = output.outputs[0].logprobs
        if logprobs:
            for _, logprob in logprobs[0].items():
                if logprob.decoded_token.strip() in valid_tks:
                    prediction = logprob.decoded_token.strip()
                    break
        all_predictions.append(prediction)

    # save predictions
    relevant_traits = {i: [] for i in dataset.index}
    for (ix, trait), answer in zip(prompt_map, all_predictions):
        if answer == "yes":
            relevant_traits[ix].append(trait)
    dataset["relevant_traits"] = [relevant_traits[i] for i in dataset.index]
    dataset.to_json(f"{DATA_PATH}/openassistant/{dataset}_{constitution}.jsonl", orient="records", lines=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/scratch/models/gemma-2-9b-base", required=False)
    parser.add_argument("--constitution", type=str, default="main")
    parser.add_argument("--dataset", type=str, default="oasst_top1", choices=["oasst_top1", "oasst2"])
    parser.add_argument("--K", type=int, default=20)
    args = parser.parse_args()
    evaluate(args.model, args.constitution, args.dataset, args.K)