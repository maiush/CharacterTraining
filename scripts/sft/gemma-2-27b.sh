source /root/mats/CharacterTraining/.env
source /root/.finetuning/bin/activate

export CUDA_HOME=/root/cuda-12.4
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export HF_HOME=/root/hf-cache
export DS_ACCELERATOR=cuda

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
openrlhf.cli.train_sft \
    --save_path ${OUTPUT_PATH}/saves \
    --ckpt_path ${OUTPUT_PATH}/ckpts \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 2 \
    --train_batch_size 48 \
    --seed 123456 \
    --zero_stage 3 \
    --adam_offload \
    --gradient_checkpointing \
    --bf16 \
    --max_epochs 5 \
    --pretrain /data/e02cf067-caa7-4a1e-8850-c5e4448d0fdf/gemma-2-27b-it \
    --dataset /root/mats/CharacterTraining/data/sft/gemma-2-27b.jsonl \
    --input_key messages \
    --apply_chat_template \
    --max_len 4096 \
    --use_wandb True \
    --wandb_project CharacterTraining \
    --wandb_run_name sft-gemma-2-27b
EOF


(
    echo "script will automatically terminate after 1 hour"
    sleep 3600  # sleep for 1 hour (3600 seconds)
    echo "time limit reached (1 hour). terminating script..."
    kill -9 $$  # kill the current script process
) &
TIMEOUT_PID=$!

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
    python -c '
import os, json, datetime
from pathlib import Path
from huggingface_hub import login, HfApi

current_time = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
OUTPUT_PATH = os.environ["OUTPUT_PATH"]
SAVE_PATH = f"{OUTPUT_PATH}/saves"

# update README.md with correct base_model before uploading
readme_path = Path(SAVE_PATH) / "README.md"
if readme_path.exists():
    with open(readme_path, "r") as f:
        content = f.read()
    # Replace the base_model line
    content = content.replace(
        "base_model: /data/e02cf067-caa7-4a1e-8850-c5e4448d0fdf/gemma-2-27b-it", 
        "base_model: google/gemma-2-27b-it"
    )
    with open(readme_path, "w") as f:
        f.write(content)
login(new_session=False)
api = HfApi()
model_name = f"lora-sft-gemma-2-27b-{current_time}"
api.create_repo(repo_id=f"maius/{model_name}")
api.upload_folder(
    folder_path=SAVE_PATH,
    repo_id=f"maius/{model_name}",
    repo_type="model"
)
'
fi

kill $TIMEOUT_PID 2>/dev/null || true