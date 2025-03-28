source /workspace/CharacterTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


cd /workspace/models
huggingface-cli download maius/gemma-2-9b-$1-1103 --local-dir ./gemma-2-9b-$1
cp -r gemma-2-9b-$1 GENERATOR


# N iterations of dpo
for i in {1..3}; do
    echo "starting DPO iteration $i of 3"

    cd /workspace/CharacterTraining/charactertraining
    # paired generations from the GENERATOR
    python dpo_generate.py \
        --N 10000 \
        --model /workspace/models/GENERATOR \
        --dataset oasst2 \
    # pairwise comparisons against the constitution from the EVALUATOR
    python dpo_evaluate.py --constitution $1

    # round of DPO
    cd /workspace
    read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
    --save_path /workspace/models/NEXT \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --gradient_checkpointing \
    --seed 123456 \
    --zero_stage 3 \
    --bf16 \
    --learning_rate 5e-7 \
    --beta 0.15 \
    --adam_betas 0.9 0.999 \
    --max_epochs 2 \
    --pretrain /workspace/models/GENERATOR \
    --dataset /workspace/CharacterTraining/data/current_eval.jsonl \
    --chosen_key chosen \
    --rejected_key rejected \
    --apply_chat_template \
    --max_len 8192 \
    --use_wandb True \
    --wandb_project CharacterTraining \
    --wandb_run_name gemma-2-9b-$1-dpo-$i
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
        --model NEXT \
        --name gemma-2-9b-$1-dpo-$i-2803
    # build the snapshot for the next generation step 
    rm -rf /workspace/models/GENERATOR
    mv /workspace/models/NEXT /workspace/models/GENERATOR
    echo "finished DPO iteration $i of 3"
done


cd /workspace/models
rm -rf gemma-2-9b-$1
rm -rf GENERATOR
rm -rf NEXT
