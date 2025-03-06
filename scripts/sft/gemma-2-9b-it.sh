source /workspace/CharacterTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


cd /workspace

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path /workspace/models/gemma-2-9b-it-lora \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 64 \
    --seed 123456 \
    --gradient_checkpointing \
    --zero_stage 0 \
    --bf16 \
    --max_epochs 5 \
    --pretrain /workspace/models/gemma-2-9b-it \
    --learning_rate 1e-4 \
    --l2 1e-2 \
    --adam_betas 0.9 0.98 \
    --lora_rank 32 \
    --lora_alpha 16 \
    --dataset /workspace/CharacterTraining/data/sft/gemma-2-9b.jsonl \
    --input_key messages \
    --apply_chat_template \
    --max_len 4096 \
    --use_wandb True \
    --wandb_project CharacterTraining \
    --wandb_run_name gemma-2-9b-it-lora
EOF


deepspeed \
--module $training_commands


# only run the following commands if the deepspeed command succeeded
if [ $? -eq 0 ]; then
    # remove wandb logs
    rm -rf /workspace/wandb
    # combine lora and base
    cd /workspace/CharacterTraining/openrlhf/openrlhf/cli
    python lora_combiner.py \
        --model_path /workspace/models/gemma-2-9b-it \
        --lora_path /workspace/models/gemma-2-9b-it-lora \
        --output_path /workspace/models/gemma-2-9b-evaluator
    # upload model
    cd /workspace/CharacterTraining/tools
    python upload_model.py --model gemma-2-9b-evaluator --name gemma-2-9b-evaluator-0603
    # upload lora
    python upload_model.py --model gemma-2-9b-it-lora --name gemma-2-9b-it-lora-0603
fi