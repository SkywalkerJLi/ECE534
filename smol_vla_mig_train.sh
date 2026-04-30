#!/bin/bash
#SBATCH --job-name=smolvla_debug
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=60:30:00
#SBATCH --partition=mig

# -----------------------------------------------------------------------------
# DEBUG RUN on MIG (10 GB GPU)
# Purpose: verify data loading, model loading, and a few training steps work
# before submitting the real job to the 40 GB partition.
# -----------------------------------------------------------------------------

# Offline mode so the compute node doesn't try to reach HuggingFace
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate environment
source /home/sl5183/ECE534/.venv/bin/activate
cd /home/sl5183/ECE534

# Print some diagnostic info
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU info:"
nvidia-smi
echo "========================================"

# Tiny batch + only 20 steps to verify the pipeline works
python lerobot/src/lerobot/scripts/lerobot_train.py \
  --policy.path=lerobot/smolvla_base \
  --policy.vlm_model_name=/scratch/gpfs/TSILVER/sl5183/ECE534/models/SmolVLM2-500M-Video-Instruct \
  --dataset.repo_id=nc8304/so101_color_augmented \
  --dataset.root=/scratch/gpfs/TSILVER/sl5183/ECE534/so101_color_augmented \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --batch_size=32 \
  --steps=20000 \
  --save_freq=5000 \
  --output_dir=/scratch/gpfs/TSILVER/sl5183/ECE534/outputs/train/smolvla_debug_32_color_aug \
  --job_name=smolvla_debug_color_aug \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false

echo "========================================"
echo "Debug job finished"
echo "========================================"
