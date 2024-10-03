#!/bin/bash
#SBATCH --job-name=h4-optillm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --output=/fsx/h4/logs/%x-%j.out
#SBATCH --err=/fsx/h4/logs/%x-%j.err
#SBATCH --partition=hopper-prod
#SBATCH --requeue
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=lewis+hfc@huggingface.co

# Print shell commands and exit immediately if a command exits with a non-zero status
set -x -e

source ~/.bashrc
conda activate optillm

# Default values
APPROACH=""
MODEL=""
DATASET_NAME=""
DATASET_SPLIT="train"
DATASET_COLUMN="prompt"
PROMPT_SUFFIX=""
NUM_SAMPLES=5
N=1
TEMPERATURE=0.7
MAX_TOKENS=2048

# Parse named command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --approach) APPROACH="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --dataset_name) DATASET_NAME="$2"; shift ;;
        --dataset_split) DATASET_SPLIT="$2"; shift ;;
        --dataset_column) DATASET_COLUMN="$2"; shift ;;
        --prompt_suffix) PROMPT_SUFFIX="${2:-}"; shift ;;
        --num_samples) NUM_SAMPLES="$2"; shift ;;
        --n) N="$2"; shift ;;
        --temperature) TEMPERATURE="$2"; shift ;;
        --max_tokens) MAX_TOKENS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# The $MODEL variable is in the form {ORG}/{MODEL_NAME}, so we extract the model name for the dataset ID:
MODEL_NAME=$(echo $MODEL | cut -d'/' -f2)

python generate_dataset.py \
  --approach $APPROACH \
  --model $MODEL \
  --dataset_name $DATASET_NAME \
  --dataset_split $DATASET_SPLIT \
  --dataset_column $DATASET_COLUMN \
  --prompt_suffix "$PROMPT_SUFFIX" \
  --num_samples $NUM_SAMPLES \
  --n $N \
  --temperature $TEMPERATURE \
  --max_tokens $MAX_TOKENS \
  --push_to_hub \
  --hub_dataset_id reliable-agents/$MODEL_NAME-optillm-completions