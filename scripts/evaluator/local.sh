source /root/mats/CharacterTraining/.env
source /root/.finetuning/bin/activate


read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path /scratch/saves \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --seed 123456 \
    --zero_stage 3 \
    --adam_offload \
    --bf16 \
    --max_epochs 1 \
    --pretrain /scratch/models/gemma-2-2b-blended \
    --learning_rate 5e-6 \
    --lora_rank 32 \
    --lora_alpha 16 \
    --dataset /root/mats/CharacterTraining/data/evaluator/gemma-2-2b.jsonl,/root/mats/CharacterTraining/data/openassistant/oasst2.jsonl \
    --dataset_probs 0.1,0.9 \
    --input_key messages \
    --apply_chat_template \
    --max_len 2048 \
    --use_wandb True \
    --wandb_project CharacterTraining \
    --wandb_run_name evaluator-gemma-2-2b-blended
EOF


deepspeed \
--module $training_commands