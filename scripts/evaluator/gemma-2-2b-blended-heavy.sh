source /workspace/CharacterTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path /workspace/gemma-2-2b-blended-heavy-evaluator \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 64 \
    --seed 123456 \
    --zero_stage 2 \
    --bf16 \
    --max_epochs 1 \
    --pretrain maius/gemma-2-2b-blended-heavy \
    --learning_rate 5e-6 \
    --lora_rank 32 \
    --lora_alpha 16 \
    --dataset /workspace/CharacterTraining/data/evaluator/gemma-2-2b.jsonl,/workspace/CharacterTraining/data/openassistant/oasst2.jsonl \
    --dataset_probs 0.1,0.9 \
    --input_key messages \
    --apply_chat_template \
    --max_len 4096 \
    --use_wandb True \
    --wandb_project CharacterTraining \
    --wandb_run_name evaluator-gemma-2-2b-blended-heavy
EOF


deepspeed \
--module $training_commands

cd /workspace/CharacterTraining/tools
python upload_model.py --model gemma-2-2b-blended-heavy-evaluator
rm -rf /workspace/scripts/evaluator/wandb