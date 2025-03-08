source /workspace/CharacterTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


cd /workspace

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path /workspace/models/gemma-2-9b-blend-sft \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 128 \
    --gradient_checkpointing \
    --zero_stage 2 \
    --bf16 \
    --max_epochs 1 \
    --pretrain /workspace/models/gemma-2-9b-blend \
    --learning_rate 1e-4 \
    --l2 1e-2 \
    --adam_betas 0.9 0.98 \
    --dataset /workspace/CharacterTraining/data/critiques/gemma-2-9b-it.jsonl \
    --input_key revision \
    --apply_chat_template \
    --max_len 8192 \
    --use_wandb True \
    --wandb_project CharacterTraining \
    --wandb_run_name gemma-2-9b-blend-sft
EOF


deepspeed \
--module $training_commands


# only run the following commands if the deepspeed command succeeded
if [ $? -eq 0 ]; then
    # remove wandb logs
    rm -rf /workspace/wandb
    # upload model
    cd /workspace/CharacterTraining/tools
    python upload_model.py --model gemma-2-9b-blend-sft --name gemma-2-9b-blend-sft-0803
fi