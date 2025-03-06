source /workspace/CharacterTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


# loop through the DPO process 5 times
for i in {1..10}; do
    echo "starting DPO iteration $i of 10"

    cd /workspace/CharacterTraining/charactertraining
    # paired generations from the GENERATOR
    python dpo_generate.py \
        --N 1000 \
        --generator /workspace/models/llama-3.1-8b-generator-snapshot \
        --evaluator /workspace/models/llama-3.1-8b-evaluator
    # ratings against the constitution from the EVALUATOR
    python dpo_evaluate.py \
        --dataset /workspace/CharacterTraining/data/dpo/current_gen.jsonl \
        --generator /workspace/models/llama-3.1-8b-generator-snapshot \
        --evaluator /workspace/models/llama-3.1-8b-evaluator

    # round of DPO
    cd /workspace
    read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
    --save_path /workspace/models/llama-3.1-8b-dpo-lora \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 128 \
    --gradient_checkpointing \
    --seed 123456 \
    --zero_stage 0 \
    --bf16 \
    --learning_rate 1e-4 \
    --beta 0.15 \
    --adam_betas 0.9 0.98 \
    --lora_rank 32 \
    --lora_alpha 16 \
    --max_epochs 1 \
    --pretrain /workspace/models/llama-3.1-8b-blend \
    --ref_pretrain /workspace/models/llama-3.1-8b-blend \
    --load_lora_adapter /workspace/models/llama-3.1-8b-generator-lora \
    --ref_load_lora_adapter /workspace/models/llama-3.1-8b-blend-lora \
    --dataset /workspace/CharacterTraining/data/dpo/current_dpo.jsonl,/workspace/CharacterTraining/data/generator/llama-3.1-8b.jsonl \
    --dataset_probs 0.8,0.2 \
    --chosen_key chosen \
    --rejected_key rejected \
    --apply_chat_template \
    --max_len 4096 \
    --use_wandb True \
    --wandb_project CharacterTraining \
    --wandb_run_name llama-3.1-8b-generator-lora-iter-$i
EOF
    deepspeed \
    --module $training_commands
    # remove wandb folder
    rm -rf /workspace/wandb
    # remove outdated generator lora
    rm -rf /workspace/models/llama-3.1-8b-generator-lora
    # add new generator lora
    mv /workspace/models/llama-3.1-8b-dpo-lora /workspace/models/llama-3.1-8b-generator-lora
    # upload the new generator lora
    cd /workspace/CharacterTraining/tools
    python upload_model.py \
        --model llama-3.1-8b-generator-lora \
        --name llama-3.1-8b-generator-lora-0603-iter-$i
    # build the snapshot for the next generation step (first remove the old snapshot)
    rm -rf /workspace/models/llama-3.1-8b-generator-snapshot
    cd /workspace/CharacterTraining/openrlhf/openrlhf/cli
    python lora_combiner.py \
        --model_path /workspace/models/llama-3.1-8b-blend \
        --lora_path /workspace/models/llama-3.1-8b-generator-lora \
        --output_path /workspace/models/llama-3.1-8b-generator-snapshot
    echo "finished DPO iteration $i of 10"
done

# upload to huggingface
cd /workspace/CharacterTraining/tools
python upload_model.py \
    --model llama-3.1-8b-generator-snapshot \
    --name llama-3.1-8b-generator-0603