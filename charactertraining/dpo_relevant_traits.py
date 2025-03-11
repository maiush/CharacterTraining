import argparse, json, random
from argparse import Namespace
import pandas as pd
import torch as t
from vllm import LLM, SamplingParams

from charactertraining.constants import DATA_PATH, CONSTITUTION_PATH


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


def random_traits(
        constitution: str,
        K: int
) -> None:
    # get constitution
    with open(f"{CONSTITUTION_PATH}/{constitution}.txt", "r") as f:
        cons = json.load(f)
    cons = pd.DataFrame(cons)
    # get current generations
    generations = pd.read_json(f"{DATA_PATH}/current_gen.jsonl", orient="records", lines=True)
    # randomly sample a category
    category = random.choice(cons["category"].unique().tolist())
    # NOTE: quick fix of stuff - might change later
    if category not in ["wisdom", "candor"]:
        category = ["thoughtfulness", "compassion", "virtue", "curiosity"]
    else:
        category = [category]
    # get all traits in this category
    traits = cons[cons["category"].isin(category)]["trait"].unique().tolist()
    # randomly sample K traits
    traits = random.sample(traits, min(K, len(traits)))
    generations["relevant_trait"] = generations["question"].apply(lambda _: random.choice(traits))
    # save relevant traits
    generations.to_json(f"{DATA_PATH}/current_gen_w_traits.jsonl", orient="records", lines=True)
        

def evaluate(
        model: str,
        constitution: str,
        K: int,
        **kwargs,
) -> None:
    assert K > 1
    # get constitution
    with open(f"{CONSTITUTION_PATH}/{constitution}.txt", "r") as f:
        cons = json.load(f)
    cons = pd.DataFrame(cons)

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

    # === GET RELEVANT TRAITS === 
    # get current generations
    generations = pd.read_json(f"{DATA_PATH}/current_gen.jsonl", orient="records", lines=True)
    relevant_traits = {}
    # for each question, find a relevant trait
    questions = generations["question"].unique().tolist()
    possible_traits = {question: cons["trait"].unique().tolist() for question in questions}
    while True:
        if len(questions) == 0: break
        # build prompts
        prompts, traits = [], []
        for question in questions:
            # choose a random trait to check
            trait = random.choice(possible_traits[question])
            traits.append(trait)
            # update possible traits
            possible_traits[question].remove(trait)
            # create prompt
            prompt = f"{few_shot_prompt}{template.format(user_message=question, trait=trait)}"
            prompts.append(prompt)
        # generate outputs
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
        # check for any relevant traits
        for question, trait, prediction in zip(questions, traits, predictions):
            if prediction == "yes":
                relevant_traits[question] = trait
                # remove from questions
                questions.remove(question)
        # remove any questions with no possible traits
        questions = [q for q in questions if len(possible_traits[q]) > 0]
    # update generations
    generations["relevant_trait"] = generations["question"].apply(lambda q: relevant_traits.get(q, ""))
    # drop rows with no relevant trait
    generations = generations[generations["relevant_trait"] != ""]
    # save relevant traits
    generations.to_json(f"{DATA_PATH}/current_gen_w_traits.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/workspace/models/llama-3.1-70b-base", required=False)
    parser.add_argument("--constitution", type=str, default="main")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--random", action="store_true", default=False)
    args = parser.parse_args()
    if args.random:
        random_traits(args.constitution, args.K)
    else:
        evaluate(args.model, args.constitution, args.K)
