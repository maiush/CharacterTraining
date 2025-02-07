'''
Build the full dataset of questions to begin the synthetic data generation process.
This includes handwritten questions, few-shot questions from Claude, and combinations of traits and questions relevant to other traits.
'''


import pandas as pd
from charactertraining.constants import DATA_PATH
from charactertraining.questions import TRAITS
from charactertraining.claude_questions import QUESTIONS as CLAUDE_QUESTIONS


data = pd.DataFrame(columns=["trait", "question", "minimal", "claude", "all", "messages"])
# load handwritten questions
for trait, questions in TRAITS.items():
    for question in questions:
        data.loc[len(data)] = {
            "trait": trait,
            "question": question,
            "minimal": True,
            "claude": False,
            "all": True,
            "messages": [{
                "role": "user",
                "content": question
            }]
        }
# load claude questions
for trait, questions in CLAUDE_QUESTIONS.items():
    for question in questions:
        data.loc[len(data)] = {
            "trait": trait,
            "question": question,
            "minimal": False,
            "claude": True,
            "all": False,
            "messages": [{
                "role": "user",
                "content": question
            }]
        }
# load combinations
all_qs = [q for _, qs in TRAITS.items() for q in qs]
for trait in TRAITS.keys():
    for q in all_qs:
        if q not in TRAITS[trait]:
            data.loc[len(data)] = {
                "trait": trait,
                "question": q,
                "minimal": False,
                "claude": False,
                "all": True,
                "messages": [{
                    "role": "user",
                    "content": q
                }]
            }

# save data
data.to_json(f"{DATA_PATH}/critiques/inputs/questions.jsonl", orient="records", lines=True)