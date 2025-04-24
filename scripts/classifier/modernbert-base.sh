source /workspace/CharacterTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


cd /workspace/CharacterTraining/charactertraining

TOKENIZERS_PARALLELISM=false WANDB_PROJECT=CharacterTraining accelerate launch --mixed_precision bf16 train_classifier.py

# only run the following commands if the deepspeed command succeeded
if [ $? -eq 0 ]; then
    # remove wandb logs
    rm -rf /workspace/CharacterTraining/charactertraining/wandb
    # upload model
    cd /workspace/CharacterTraining/tools
    python upload_model.py --model modernbert-base-classifier --name modernbert-base-classifier-0424
fi