source /root/mats/CharacterTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


cd /root/mats/CharacterTraining/charactertraining

python dpo_generate.py \
    --N 100 \
    --generator /scratch/ct_models/gemma-2-2b-sft \
    --evaluator /scratch/ct_models/gemma-2-2b-it


python dpo_evaluate.py \
    --dataset /root/mats/CharacterTraining/data/dpo/current_gen.jsonl \
    --generator /scratch/ct_models/gemma-2-2b-sft \
    --evaluator /scratch/ct_models/gemma-2-2b-it
