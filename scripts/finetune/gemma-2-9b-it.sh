source /workspace/CharacterTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


# run sft
cd /workspace/CharacterTraining/scripts/sft
./gemma-2-9b-it.sh $1
sleep 10
# prepare for dpo
cd /workspace/models
rm -rf GENERATOR
cp -r gemma-2-9b-it-sft-$1 GENERATOR
sleep 10
# run dpo
cd /workspace/CharacterTraining/scripts/dpo
./gemma-2-9b-it.sh $1