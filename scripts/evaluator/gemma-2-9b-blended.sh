source /workspace/CharacterTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path /workspace/gemma-2-9b-blended-evaluator \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 64 \
    --seed 123456 \
    --gradient_checkpointing \
    --zero_stage 2 \
    --bf16 \
    --max_epochs 1 \
    --pretrain maius/gemma-2-9b-blended \
    --learning_rate 5e-6 \
    --lora_rank 32 \
    --lora_alpha 16 \
    --dataset /workspace/CharacterTraining/data/evaluator/gemma-2-9b.jsonl,/workspace/CharacterTraining/data/openassistant/oasst2.jsonl \
    --dataset_probs 0.2,0.8 \
    --input_key messages \
    --apply_chat_template \
    --max_len 4096 \
    --use_wandb True \
    --wandb_project CharacterTraining \
    --wandb_run_name evaluator-gemma-2-9b-blended
EOF


deepspeed \
--module $training_commands

# only run the following commands if the deepspeed command succeeded
if [ $? -eq 0 ]; then
    cd /workspace/CharacterTraining/tools
    python upload_model.py --model gemma-2-9b-blended-evaluator
    rm -rf /workspace/CharacterTraining/scripts/evaluator/wandb
fi