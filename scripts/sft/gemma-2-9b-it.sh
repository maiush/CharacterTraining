source /workspace/CharacterTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


cd /workspace

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path /workspace/models/gemma-2-9b-it-sft-$1 \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 128 \
    --gradient_checkpointing \
    --zero_stage 2 \
    --bf16 \
    --max_epochs 1 \
    --pretrain /workspace/models/gemma-2-9b-it \
    --learning_rate 5e-6 \
    --adam_betas 0.9 0.98 \
    --dataset /workspace/CharacterTraining/data/acr/gemma-2-9b-it/$1.jsonl \
    --input_key revision \
    --apply_chat_template \
    --max_len 8192 \
    --use_wandb True \
    --wandb_project CharacterTraining \
    --wandb_run_name gemma-2-9b-it-sft-$1 \
    --seed 123456
EOF


deepspeed \
--module $training_commands


# only run the following commands if the deepspeed command succeeded
if [ $? -eq 0 ]; then
    # remove wandb logs
    rm -rf /workspace/wandb
    # upload model
    cd /workspace/CharacterTraining/tools
    python upload_model.py --model gemma-2-9b-it-sft-$1 --name gemma-2-9b-it-sft-$1-0504
fi