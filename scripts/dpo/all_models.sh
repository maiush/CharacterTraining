#!/bin/bash

traits=("humor" "sarcasm" "remorse" "wisdom" "candor" "lfh")

# loop through each trait and run the download_and_train.sh script
for trait in "${traits[@]}"; do
    echo "=========================================="
    echo "starting training for trait: $trait"
    echo "=========================================="
    
    # run the download_and_train.sh script with the current trait
    bash /workspace/CharacterTraining/scripts/dpo/download_and_train.sh "$trait"
    
    # check if the script executed successfully
    if [ $? -eq 0 ]; then
        echo "=========================================="
        echo "Successfully completed training for trait: $trait"
        echo "=========================================="
    else
        echo "=========================================="
        echo "ERROR: Training failed for trait: $trait"
        echo "=========================================="
        exit 1
    fi
    
    sleep 5
done

echo "all training runs completed!"
