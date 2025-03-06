import argparse, json, random
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from charactertraining.dpo_generate import gen_args
from charactertraining.constants import CONSTITUTION_PATH

evaluator_template = """\
Your task is to compare the extent to which two sample AI assistant responses to a given human user message exhibit a particular principle.

=== Begin Principle ===
{principle}
=== End Principle ===

=== Begin User Message ===
{question}
=== End User Message ===

=== Begin Response A ===
{choice1}
=== End Response A ===

=== Begin Response B ===
{choice2}
=== End Response B ===

Which response is more aligned with the principle? Select either A or B, and respond with your chosen letter only.\
"""

valid_tks = ["A", "B"]


def evaluate(
        GENERATE_PATH: str,
        generator: str,
        evaluator: str,
        **kwargs,
) -> None:
    # get generations
    dataset = pd.read_json(GENERATE_PATH, orient="records", lines=True)

    # load constitutional principles
    with open(f"{CONSTITUTION_PATH}/main.txt", "r") as f:
        con = json.load(f)
    con = pd.DataFrame(con)

    def gen_eval_prompts(revisions, main_question, choice1, choice2, reverse=False):
        if reverse:
            choice1, choice2 = choice2, choice1
        prompts = []
        for principle in con["trait"].unique():
            main_q = evaluator_template.format(
                principle=principle,
                question=main_question,
                choice1=choice1,
                choice2=choice2
            )
            fs_prompt = []
            questions = con.loc[con["trait"] == principle, "questions"].item()
            formatted_questions, answers = [], []
            for question in questions:
                sample = revisions[revisions["question"] == question].sample(1)
                A, B = sample["initial"].item(), sample["revisions"].item()
                p = random.random()
                fq = evaluator_template.format(
                    principle=principle,
                    question=question,
                    choice1=A if p < 0.5 else B,
                    choice2=B if p < 0.5 else A
                )
                answer = "B" if p < 0.5 else "A"
                formatted_questions.append(fq)
                answers.append(answer)
            for q, a in zip(formatted_questions, answers):
                fs_prompt.append(
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                )
            fs_prompt.append({"role": "user", "content": main_q})
            prompts.append(fs_prompt)        
        return prompts

    # gen inference args
    args = gen_args(generator, evaluator, max_new_tokens=1, **kwargs)
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
        model=args.evaluator,
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

    # create one huge list of few-shot prompts
    model_name = evaluator.split("/")[-1]
    idx = model_name.index("b-")
    model_name = model_name[:idx+1]
    revisions_path = f"/workspace/CharacterTraining/data/generator/{model_name}.jsonl"
    revisions = pd.read_json(revisions_path, orient="records", lines=True)
    all_prompts = []
    row_indices = []    
    for idx, row in dataset.iterrows():
        question = row["question"]
        choice1 = row["choice1"]
        choice2 = row["choice2"]
        prompts = gen_eval_prompts(revisions, question, choice1, choice2, reverse=False)
        # add the row index K times (once for each prompt)
        row_indices.extend([idx] * len(prompts))
        all_prompts.extend(prompts)
    
    # preprocess all prompts at once
    all_prompts = tokenizer.apply_chat_template(
        all_prompts,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # generate all outputs at once
    all_outputs = llm.generate(all_prompts, sampling_params)
    
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
    
    # calculate winners by taking majority vote for each group of K predictions
    winner_list = []
    for i in range(0, len(dataset)):
        # find all predictions for this row
        row_predictions = []
        for j in range(len(row_indices)):
            if row_indices[j] == i:
                row_predictions.append(all_predictions[j])
        
        total_A = sum(1 for x in row_predictions if x == "A")
        total_B = sum(1 for x in row_predictions if x == "B")
        
        if total_A > total_B:
            winner = 1
        elif total_A < total_B:
            winner = 2
        else:
            winner = -1
        
        winner_list.append(winner)
    
    dataset["winner"] = winner_list
    # save results
    outpath = GENERATE_PATH.replace("gen.jsonl", "eval.jsonl")
    dataset.to_json(outpath, orient="records", lines=True)
    # build dpo dataset
    dataset = dataset[dataset["winner"] != -1]
    dataset["chosen"] = dataset.apply(
        lambda row: row["messages"] + [{"role": "assistant", "content": row[f'choice{row.winner}']}], axis=1
    )
    dataset["rejected"] = dataset.apply(
        lambda row: row["messages"] + [{"role": "assistant", "content": row[f'choice{2 if row.winner == 1 else 1}']}], axis=1
    )
    # save dpo dataset
    outpath = GENERATE_PATH.replace("gen.jsonl", "dpo.jsonl")
    dataset.to_json(outpath, orient="records", lines=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/root/CharacterTraining/data/dpo/current_gen.jsonl")
    parser.add_argument("--generator", type=str, default="/scratch/ct_models/gemma-2-2b-sft")
    parser.add_argument("--evaluator", type=str, default="/scratch/ct_models/gemma-2-2b-it")
    args = parser.parse_args()
    evaluate(args.dataset, args.generator, args.evaluator)