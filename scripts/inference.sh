export DATA_PATH="/root/mats/CharacterTraining/data"
source /root/.bashrc
source /root/finetuning/bin/activate

export CUDA_HOME=/root/cuda-12.6
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

set -x

read -r -d '' inference_args <<EOF
openrlhf.cli.batch_inference \
   --eval_task generate_vllm \
   --zero_stage 3 \
   --bf16 \
   --flash_attn \
   --micro_batch_size 8 \
   --seed 123456 \
   --pretrain meta-llama/Llama-3.1-8B-Instruct \
   --dataset ${DATA_PATH}/critiques/inputs/llama/rephrased_large.jsonl \
   --input_key question \
   --apply_chat_template \
   --max_len 4096 \
   --output_path ${DATA_PATH}/critiques/outputs/llama/rephrased_large.jsonl
   --max_new_tokens 4096
EOF

python -m $inference_args
