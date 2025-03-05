source /workspace/CharacterTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path /workspace/models/gemma-2-9b-lora \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 64 \
    --seed 123456 \
    --gradient_checkpointing \
    --zero_stage 0 \
    --bf16 \
    --max_epochs 1 \
    --pretrain maius/gemma-2-9b-blend \
    --learning_rate 1e-4 \
    --l2 1e-2 \
    --lora_rank 32 \
    --lora_alpha 16 \
    --dataset /workspace/CharacterTraining/data/evaluator/gemma-2-9b.jsonl \
    --input_key messages \
    --apply_chat_template \
    --max_len 4096 \
    --use_wandb True \
    --wandb_project CharacterTraining \
    --wandb_run_name evaluator-gemma-2-9b-lora
EOF


deepspeed \
--module $training_commands

# only run the following commands if the deepspeed command succeeded
if [ $? -eq 0 ]; then
    cd /workspace/CharacterTraining/tools
    python upload_model.py --model gemma-2-9b-lora
    rm -rf /workspace/CharacterTraining/scripts/evaluator/wandb
fi