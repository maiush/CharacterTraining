source /workspace/CharacterTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


cd /workspace/CharacterTraining/charactertraining

python gen_questions.py --constitution wisdom --model /workspace/models/llama-3.1-70b-base

python gen_questions.py --constitution candor --model /workspace/models/llama-3.1-70b-base

python gen_questions.py --constitution humor --model /workspace/models/llama-3.1-70b-base

python gen_questions.py --constitution sarcasm --model /workspace/models/llama-3.1-70b-base

python gen_questions.py --constitution remorse --model /workspace/models/llama-3.1-70b-base