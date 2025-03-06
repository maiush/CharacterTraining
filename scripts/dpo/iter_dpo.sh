source /workspace/CharacterTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


(
    echo "script will automatically terminate after 3 hours"
    sleep 10800  # sleep for 3 hours (10800 seconds)
    echo "time limit reached (3 hours). terminating pod..."
    runpodctl remove pod 2dwxpfnyhmi53k  # terminate the pod
) &
TIMEOUT_PID=$!


# Loop through the DPO process 5 times
for i in {1..5}; do
    echo "starting DPO iteration $i of 5"
    cd /workspace/CharacterTraining/charactertraining

    # paired generations from the GENERATOR
    python dpo_generate.py \
        --N 5000 \
        --generator /workspace/models/gemma-2-9b-generator \
        --evaluator /workspace/models/gemma-2-9b-evaluator

    # ratings against the constitution from the EVALUATOR
    python dpo_evaluate.py \
        --dataset /workspace/CharacterTraining/data/dpo/current_gen.jsonl \
        --generator /workspace/models/gemma-2-9b-generator \
        --evaluator /workspace/models/gemma-2-9b-evaluator

    # round of DPO
    cd /workspace
    read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
    --save_path /workspace/models/gemma-2-9b-dpo \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 128 \
    --gradient_checkpointing \
    --seed 123456 \
    --zero_stage 0 \
    --bf16 \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --lora_alpha 16 \
    --max_epochs 3 \
    --pretrain /workspace/models/gemma-2-9b-generator \
    --dataset /workspace/CharacterTraining/data/dpo/current_dpo.jsonl,/workspace/CharacterTraining/data/generator/gemma-2-9b.jsonl \
    --dataset_probs 0.9,0.1 \
    --chosen_key chosen \
    --rejected_key rejected \
    --apply_chat_template \
    --max_len 4096 \
    --use_wandb True \
    --wandb_project CharacterTraining \
    --wandb_run_name gemma-2-9b-dpo-iter-$i
EOF

    deepspeed \
    --module $training_commands

    # remove wandb folder
    rm -rf /workspace/wandb
    # remove outdated generator
    rm -rf /workspace/models/gemma-2-9b-generator
    # merge lora to build new generator
    cd /workspace/CharacterTraining/openrlhf/openrlhf/cli
    python lora_combiner.py \
        --model_path /workspace/models/gemma-2-9b-generator-snapshot \
        --lora_path /workspace/models/gemma-2-9b-dpo \
        --output_path /workspace/models/gemma-2-9b-generator
    # remove lora directory
    rm -rf /workspace/models/gemma-2-9b-dpo
    # upload the final model
    cd /workspace/CharacterTraining/tools
    python upload_model.py \
        --model gemma-2-9b-generator \
        --name gemma-2-9b-generator-iter-$i
    
    echo "finished DPO iteration $i of 5"
done

# when done, terminate the pod
runpodctl remove pod 2dwxpfnyhmi53k