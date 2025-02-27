source /root/mats/CharacterTraining/.env
source /root/.finetuning/bin/activate

export CUDA_HOME=/root/cuda-12.4
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export HF_HOME=/root/hf-cache
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export OUTPUT_PATH=$1
export NNODES=$2
export N_PROC=$3
export MASTER_ADDR=$4
export MASTER_PORT=$5
export RANK=$6
export IP_TO_RANK_MAPPING=$7
export MODEL=$8

echo "output_path: $OUTPUT_PATH"
echo "n_nodes: $NNODES"
echo "n_proc: $N_PROC"
echo "master_addr: $MASTER_ADDR"
echo "master_port: $MASTER_PORT"
echo "rank: $RANK"
echo "ip_to_rank_mapping: $IP_TO_RANK_MAPPING"
echo "model: $MODEL"

cd /root/mats/CharacterTraining/charactertraining
python critique.py --model $MODEL --outpath $OUTPUT_PATH

if [ $? -eq 0 ]; then
    sleep 2m
fi