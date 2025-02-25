source /root/mats/CharacterTraining/.env
source /root/.finetuning/bin/activate

export CUDA_HOME=/root/cuda-12.4
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export HF_HOME=/root/hf-cache

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path /scratch/saves \
    --ckpt_path /scratch/ckpts \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --seed 123456 \
    --zero_stage 3 \
    --adam_offload \
    --bf16 \
    --max_epochs 10 \
    --pretrain /root/hf-cache/gemma-2-2b-it \
    --dataset /root/mats/CharacterTraining/data/sft/gemma-2-2b.jsonl \
    --input_key messages \
    --apply_chat_template \
    --max_len 4096 \
    --use_wandb True \
    --wandb_project CharacterTraining \
    --wandb_run_name sft-gemma-2-2b
EOF


deepspeed \
--module $training_commands

if [ $? -eq 0 ]; then
    sleep 2m
fi