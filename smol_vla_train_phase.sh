#!/bin/bash
#SBATCH --job-name=smolvla_phase_train
#SBATCH --output=logs/phase_train_%j.out
#SBATCH --error=logs/phase_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=50:00:00
#SBATCH --constraint="intel&gpu40"

# -----------------------------------------------------------------------------
# PHASE-SPLIT TRAINING on 40 GB GPU (A100-40GB)
# Dataset:   so101_phase_split  (built by build_phase_dataset.py)
# Episodes:  first 240 sub-episodes = 80 unique source trajectories × 3 prompts
#            (approach / carry / full), no color-aug copies.
# 20k steps @ batch size 64.  Adjust --time if you need longer.
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

# Sub-episode layout in so101_phase_split:
#   new_ep_idx = 3 * src_ep + {0:approach, 1:carry, 2:full}
# Indices 0..239 cover source episodes 0..79, all three prompts each.
EPISODE_SUBSET=$(python -c 'print(list(range(240)))')

python lerobot/src/lerobot/scripts/lerobot_train.py \
  --policy.path=lerobot/smolvla_base \
  --policy.vlm_model_name=/scratch/gpfs/TSILVER/sl5183/ECE534/models/SmolVLM2-500M-Video-Instruct \
  --dataset.repo_id=local/so101_phase_split \
  --dataset.root=/scratch/gpfs/TSILVER/sl5183/ECE534/so101_phase_split \
  --dataset.episodes="$EPISODE_SUBSET" \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --batch_size=64 \
  --steps=20000 \
  --save_freq=5000 \
  --output_dir=/scratch/gpfs/TSILVER/sl5183/ECE534/outputs/train/my_smolvla_phase_split \
  --job_name=my_smolvla_training_phase_split \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false

echo "========================================"
echo "Phase-split training job finished"
echo "========================================"
