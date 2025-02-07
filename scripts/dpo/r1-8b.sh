source /root/.bashrc
source /root/CharacterTraining/.env
source /root/finetuning/bin/activate

export CUDA_HOME=/root/cuda-12.6
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export HF_HOME=/root/hf-cache

export OUTPUT_PATH=$1
export NNODES=$2
export N_PROC=$3
export MASTER_ADDR=$4
export MASTER_PORT=$5
export RANK=$6
export IP_TO_RANK_MAPPING=$7

echo "output_path: $OUTPUT_PATH"
echo "n_nodes: $NNODES"
echo "n_proc: $N_PROC"
echo "master_addr: $MASTER_ADDR"
echo "master_port: $MASTER_PORT"
echo "rank: $RANK"
echo "ip_to_rank_mapping: $IP_TO_RANK_MAPPING"

python -c '
import sys, json, ast, os
n_proc = int(os.environ["N_PROC"])
mapping = str(os.environ["IP_TO_RANK_MAPPING"])[2:-2]
mapping = mapping.split(",")
mapping = {int(rank): ip for rank, ip in [item.split(":") for item in mapping]}
hostfile_path = "/root/mats/CharacterTraining/isc/hostfile"
with open(hostfile_path, "w") as f:
    for rank, ip in mapping.items():
        line = f"{ip} slots={n_proc}\n"
        f.write(line)
print(f"hostfile created at: {hostfile_path}")
'

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
    --save_path ${OUTPUT_PATH}/saves \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 128 \
    --gradient_checkpointing \
    --seed 123456 \
    --local_rank $RANK \
    --zero_stage 3 \
    --adam_offload \
    --bf16 \
    --learning_rate 5e-7 \
    --max_epochs 1 \
    --pretrain /root/mats/CharacterTraining/saves/sft/r1-8b \
    --dataset /root/mats/CharacterTraining/data/dpo/r1-8b.jsonl \
    --chosen_key chosen \
    --rejected_key rejected \
    --apply_chat_template \
    --max_len 2048 \
    --use_wandb True \
    --wandb_project CharacterTraining \
    --wandb_run_name r1-8b-dpo
EOF


nvidia-smi

deepspeed \
--no_ssh \
--hostfile /root/mats/CharacterTraining/isc/hostfile \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
--num_nodes $NNODES \
--num_gpus $N_PROC \
--node_rank $RANK \
--module $training_commands

if [ $? -eq 0 ]; then
    sleep 10m
fi
