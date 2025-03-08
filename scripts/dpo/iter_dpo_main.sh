source /workspace/CharacterTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


# loop through the DPO process 5 times
for i in {1..5}; do
    echo "starting DPO iteration $i of 5"

    cd /workspace/CharacterTraining/charactertraining
    # answer-critique-revise
    python critique.py --model /workspace/models/gemma-2-9b-GENERATOR --outpath /workspace/CharacterTraining/data/current_dpo.jsonl

    # round of DPO
    cd /workspace
    read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
    --save_path /workspace/models/gemma-2-9b-next \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 128 \
    --gradient_checkpointing \
    --seed 123456 \
    --zero_stage 3 \
    --adam_offload \
    --bf16 \
    --learning_rate 1e-4 \
    --beta 0.15 \
    --adam_betas 0.9 0.98 \
    --max_epochs 1 \
    --pretrain /workspace/models/gemma-2-9b-GENERATOR \
    --ref_pretrain /workspace/models/gemma-2-9b-prev \
    --dataset /workspace/CharacterTraining/data/current_dpo.jsonl \
    --chosen_key revision \
    --rejected_key initial \
    --apply_chat_template \
    --max_len 4096 \
    --use_wandb True \
    --wandb_project CharacterTraining \
    --wandb_run_name gemma-2-9b-iter-$i
EOF
    deepspeed \
    --module $training_commands
    if [ $? -ne 0 ]; then
        echo "error: deepspeed command failed in iteration $i"
        exit 1
    fi
    # remove wandb folder
    rm -rf /workspace/wandb
    # upload the new model
    cd /workspace/CharacterTraining/tools
    python upload_model.py \
        --model gemma-2-9b-next \
        --name gemma-2-9b-next-0803-iter-$i
    # build the snapshot for the next generation step 
    rm -rf /workspace/models/gemma-2-9b-prev
    mv /workspace/models/gemma-2-9b-GENERATOR /workspace/models/gemma-2-9b-prev
    mv /workspace/models/gemma-2-9b-next /workspace/models/gemma-2-9b-GENERATOR
    echo "finished DPO iteration $i of 5"
done