#!/bin/bash
#SBATCH --job-name=smolvla_phase
#SBATCH --output=logs/phase_%j.out
#SBATCH --error=logs/phase_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=60:30:00
#SBATCH --partition=mig

# -----------------------------------------------------------------------------
# PHASE-SPLIT TRAINING on MIG (10 GB GPU)
# Dataset:   so101_phase_split  (built by build_phase_dataset.py)
# Episodes:  first 1200 sub-episodes = 80 unique source trajectories × 3 prompts x 5 augmentations
#            (approach / carry / full), no color-aug copies.
# Goal:      verify SmolVLA conditions on the textual prompt to switch
#            between the three sub-trajectories.
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
EPISODE_SUBSET=$(python -c 'print(list(range(1200)))')

python lerobot/src/lerobot/scripts/lerobot_train.py \
  --policy.path=lerobot/smolvla_base \
  --policy.vlm_model_name=/scratch/gpfs/TSILVER/sl5183/ECE534/models/SmolVLM2-500M-Video-Instruct \
  --dataset.repo_id=local/so101_phase_split \
  --dataset.root=/scratch/gpfs/TSILVER/sl5183/ECE534/so101_phase_split \
  --dataset.episodes="$EPISODE_SUBSET" \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --batch_size=32 \
  --steps=20000 \
  --save_freq=5000 \
  --output_dir=/scratch/gpfs/TSILVER/sl5183/ECE534/outputs/train/smolvla_phase_split_1200 \
  --job_name=smolvla_phase_split \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false

echo "========================================"
echo "Phase-split training job finished"
echo "========================================"
