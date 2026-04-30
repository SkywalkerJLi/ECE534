#!/bin/bash
#SBATCH --job-name=smolvla_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=40:00:00
#SBATCH --constraint="intel&gpu40"

# -----------------------------------------------------------------------------
# FULL TRAINING RUN on 40 GB GPU (A100-40GB, 50% node — usually shorter queue)
# 20k steps @ batch size 64. Adjust --time if you need longer.
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

python lerobot/src/lerobot/scripts/lerobot_train.py \
  --policy.path=lerobot/smolvla_base \
  --policy.vlm_model_name=/scratch/gpfs/TSILVER/sl5183/ECE534/models/SmolVLM2-500M-Video-Instruct \
  --dataset.repo_id=nc8304/so101_color_augmented \
  --dataset.root=/scratch/gpfs/TSILVER/sl5183/ECE534/so101_color_augmented \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --batch_size=64 \
  --steps=20000 \
  --save_freq=5000 \
  --output_dir=/scratch/gpfs/TSILVER/sl5183/ECE534/outputs/train/my_smolvla_color_aug \
  --job_name=my_smolvla_training_color_aug \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false

echo "========================================"
echo "Training job finished"
echo "========================================"
