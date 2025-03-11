source /workspace/CharacterTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


cd /workspace/CharacterTraining/charactertraining

python acr.py --model /workspace/models/gemma-2-9b-it --dataset /workspace/CharacterTraining/data/reward_modelling/wisdom.jsonl --outpath /workspace/CharacterTraining/data/reward_modelling/wisdom-gemma-2-9b-it.jsonl

python acr.py --model /workspace/models/gemma-2-9b-it --dataset /workspace/CharacterTraining/data/reward_modelling/candor.jsonl --outpath /workspace/CharacterTraining/data/reward_modelling/candor-gemma-2-9b-it.jsonl

python acr.py --model /workspace/models/gemma-2-9b-it --dataset /workspace/CharacterTraining/data/reward_modelling/humor.jsonl --outpath /workspace/CharacterTraining/data/reward_modelling/humor-gemma-2-9b-it.jsonl

python acr.py --model /workspace/models/gemma-2-9b-it --dataset /workspace/CharacterTraining/data/reward_modelling/sarcasm.jsonl --outpath /workspace/CharacterTraining/data/reward_modelling/sarcasm-gemma-2-9b-it.jsonl

python acr.py --model /workspace/models/gemma-2-9b-it --dataset /workspace/CharacterTraining/data/reward_modelling/remorse.jsonl --outpath /workspace/CharacterTraining/data/reward_modelling/remorse-gemma-2-9b-it.jsonl

