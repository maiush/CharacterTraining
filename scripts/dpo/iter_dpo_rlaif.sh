source /workspace/CharacterTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


# loop through the DPO process 5 times
for i in {1..10}; do
    echo "starting DPO iteration $i of 10"

    cd /workspace/CharacterTraining/charactertraining
    # paired generations from the GENERATOR
    python dpo_generate.py \
        --N 5000 \
        --model /workspace/models/gemma-2-9b-GENERATOR \
        --dataset oasst_top1 \
    # determine relevant traits from the EVALUATOR
    python dpo_relevant_traits.py --K 10 --random
    # pairwise comparisons against the constitution from the EVALUATOR
    python dpo_evaluate.py

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
    --learning_rate 5e-7 \
    --beta 0.15 \
    --adam_betas 0.9 0.98 \
    --max_epochs 1 \
    --pretrain /workspace/models/gemma-2-9b-GENERATOR \
    --dataset /workspace/CharacterTraining/data/current_dpo.jsonl \
    --chosen_key chosen \
    --rejected_key rejected \
    --apply_chat_template \
    --max_len 8192 \
    --use_wandb True \
    --wandb_project CharacterTraining \
    --wandb_run_name gemma-2-9b-rlaif-random-iter-$i
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
        --name gemma-2-9b-rlaif-random-1003-iter-$i
    # build the snapshot for the next generation step 
    rm -rf /workspace/models/gemma-2-9b-GENERATOR
    mv /workspace/models/gemma-2-9b-next /workspace/models/gemma-2-9b-GENERATOR
    echo "finished DPO iteration $i of 10"
done
