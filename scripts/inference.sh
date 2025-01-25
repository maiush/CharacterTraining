export DATA_PATH="/gws/nopw/j04/ai4er/users/maiush/CharacterTraining/data"

set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.batch_inference \
   --eval_task generate_vllm \
   --zero_stage 0 \
   --bf16 \
   --flash_attn \
   --micro_batch_size 16 \
   --seed 123456 \
   --pretrain google/gemma-2-2b-it \
   --dataset ${DATA_PATH}/critiques/inputs/questions.jsonl \
   --input_key question \
   --apply_chat_template \
   --max_len 2048 \
   --output_path ${DATA_PATH}/critiques/outputs/questions.jsonl
   --max_new_tokens 1024 
EOF

deepspeed --module $training_commands